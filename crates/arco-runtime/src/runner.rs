//! Running a set of nodes over one bus.

use std::sync::Arc;
use std::time::Duration;

use arco_core::Error;

use crate::bus::Bus;
use crate::node::{Handle, Node, Outcome};

/// A set of nodes sharing one bus.
///
/// The Python runner also owned the display loop. That half stays in
/// Python, per deviation A-06, so this crate carries no display
/// dependency and a node is just work on a thread.
#[derive(Debug)]
pub struct Runner {
    bus: Arc<Bus>,
    handles: Vec<Handle>,
}

impl Runner {
    /// Builds a runner over a bus of the given queue capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            bus: Arc::new(Bus::new(capacity)),
            handles: Vec::new(),
        }
    }

    /// The shared bus, for wiring nodes before they start.
    #[must_use]
    pub fn bus(&self) -> &Arc<Bus> {
        &self.bus
    }

    /// Starts a node and keeps its handle.
    ///
    /// # Errors
    ///
    /// As [`Handle::start`].
    pub fn spawn(&mut self, node: Box<dyn Node>) -> Result<(), Error> {
        self.handles.push(Handle::start(node)?);
        Ok(())
    }

    /// How many nodes are still running.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.handles
            .iter()
            .filter(|handle| handle.is_running())
            .count()
    }

    /// Every node's name and current outcome.
    #[must_use]
    pub fn outcomes(&self) -> Vec<(String, Outcome)> {
        self.handles
            .iter()
            .map(|handle| (handle.name().to_owned(), handle.outcome()))
            .collect()
    }

    /// Every node that did not complete cleanly.
    ///
    /// The Python runner had no equivalent, because a node that crashed
    /// left no trace. This is what makes a pipeline failure visible to
    /// whoever started it.
    #[must_use]
    pub fn failures(&self) -> Vec<(String, Outcome)> {
        self.outcomes()
            .into_iter()
            .filter(|(_, outcome)| matches!(outcome, Outcome::Failed(_) | Outcome::Panicked(_)))
            .collect()
    }

    /// Asks every node to stop, then waits for each.
    ///
    /// Asking all of them first matters: stopping them one at a time lets
    /// each run for as long as the ones before it took to shut down.
    pub fn stop_all(&mut self, timeout: Option<Duration>) -> Vec<(String, Outcome)> {
        for handle in &self.handles {
            handle.request_stop();
        }
        self.handles
            .iter_mut()
            .map(|handle| (handle.name().to_owned(), handle.stop(timeout)))
            .collect()
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        // A runner going out of scope must not leave threads publishing
        // into a bus nobody reads.
        for handle in &self.handles {
            handle.request_stop();
        }
    }
}
