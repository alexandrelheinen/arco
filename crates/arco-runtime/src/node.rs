//! Pipeline nodes and their lifecycle.
//!
//! A node is a unit of work that runs on its own thread, publishes to the
//! bus, and stops when asked. The Python version swallowed every exception
//! its node raised, so a node that crashed on its first line was
//! indistinguishable from one that ran to completion. Here the outcome is
//! recorded and the runner can report it.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arco_core::Error;

/// How a node's run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Still running, or never started.
    Running,
    /// Returned without error.
    Completed,
    /// Returned an error.
    Failed(String),
    /// Panicked.
    ///
    /// Distinct from [`Outcome::Failed`] because a panic means a broken
    /// invariant rather than a condition the node anticipated.
    Panicked(String),
}

/// The handle a running node uses to notice it should stop.
#[derive(Debug, Clone)]
pub struct Control {
    stop: Arc<AtomicBool>,
}

impl Control {
    /// Whether the node has been asked to stop.
    ///
    /// A long-running node checks this every iteration. One that never
    /// checks it cannot be stopped, which is why [`Node::run`] documents
    /// the obligation rather than leaving it to be discovered.
    #[must_use]
    pub fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

/// A unit of pipeline work.
pub trait Node: Send {
    /// The node's name, used in reports and as the thread name.
    fn name(&self) -> &str;

    /// Runs the node once.
    ///
    /// A node that loops must check [`Control::stop_requested`] on every
    /// pass and return when it is set. A node that ignores it will be
    /// waited on until it returns of its own accord, which for an infinite
    /// loop is never.
    ///
    /// # Errors
    ///
    /// Returns whatever the work failed with. The runner records it and
    /// keeps the other nodes running.
    fn run(&mut self, control: &Control) -> Result<(), Error>;
}

/// A started node, and the means to stop it.
#[derive(Debug)]
pub struct Handle {
    name: String,
    stop: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    outcome: Arc<Mutex<Outcome>>,
    thread: Option<JoinHandle<()>>,
}

impl Handle {
    /// Starts `node` on its own thread.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConflictingArguments`] when the thread cannot be
    /// spawned, which in practice means the process is out of threads.
    pub fn start(mut node: Box<dyn Node>) -> Result<Self, Error> {
        let name = node.name().to_owned();
        let stop = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let outcome = Arc::new(Mutex::new(Outcome::Running));

        let control = Control {
            stop: Arc::clone(&stop),
        };
        let thread_outcome = Arc::clone(&outcome);
        let thread_finished = Arc::clone(&finished);

        let thread = thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                // A panic in one node must not take the process down with
                // it, so it is caught, recorded, and reported rather than
                // unwinding out of the thread.
                let result = catch_unwind(AssertUnwindSafe(|| node.run(&control)));
                let recorded = match result {
                    Ok(Ok(())) => Outcome::Completed,
                    Ok(Err(failure)) => Outcome::Failed(failure.to_string()),
                    Err(panic) => Outcome::Panicked(describe_panic(&panic)),
                };
                if let Ok(mut slot) = thread_outcome.lock() {
                    *slot = recorded;
                }
                thread_finished.store(true, Ordering::Release);
            })
            .map_err(|failure| Error::ConflictingArguments {
                message: format!("cannot start node {name}: {failure}"),
            })?;

        Ok(Self {
            name,
            stop,
            finished,
            outcome,
            thread: Some(thread),
        })
    }

    /// The node's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the node's thread is still running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.thread.is_some() && !self.finished.load(Ordering::Acquire)
    }

    /// How the run ended, or [`Outcome::Running`] if it has not.
    #[must_use]
    pub fn outcome(&self) -> Outcome {
        self.outcome
            .lock()
            .map_or(Outcome::Running, |slot| slot.clone())
    }

    /// Asks the node to stop, without waiting.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Asks the node to stop and waits for it.
    ///
    /// With `timeout` set, gives up waiting after that long and leaves the
    /// thread running, which is what the Python `join(timeout=...)` did.
    /// Returns the outcome, which is [`Outcome::Running`] on a timeout.
    #[must_use]
    pub fn stop(&mut self, timeout: Option<Duration>) -> Outcome {
        self.request_stop();

        let Some(thread) = self.thread.take() else {
            return self.outcome();
        };

        if let Some(limit) = timeout {
            // The clock ban in clippy.toml exists so a control loop takes
            // its interval as an argument and stays reproducible. A
            // shutdown deadline is wall-clock by definition and is not on
            // any reproducible path.
            #[expect(
                clippy::disallowed_methods,
                reason = "a shutdown deadline is wall-clock by nature, unlike a control interval"
            )]
            let deadline = Instant::now() + limit;
            while !self.finished.load(Ordering::Acquire) {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "polling the same shutdown deadline"
                )]
                let expired = Instant::now() >= deadline;
                if expired {
                    // The thread outlives the handle deliberately: forcing
                    // it down is not available, and pretending it stopped
                    // would be worse than saying it did not.
                    self.thread = Some(thread);
                    return Outcome::Running;
                }
                thread::sleep(Duration::from_millis(1));
            }
        }

        if thread.join().is_err() {
            // The body already caught its own panic, so reaching here
            // means the recording itself failed.
            return Outcome::Panicked("node thread failed to join".to_owned());
        }
        self.outcome()
    }
}

/// Turns a caught panic payload into something printable.
fn describe_panic(panic: &Box<dyn core::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "panicked with a non-string payload".to_owned()
    }
}
