// Copyright 2026 alexandre
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Nodes start, stop when asked, and report how they ended.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arco_core::Error;
use arco_runtime::bus::Bus;
use arco_runtime::node::{Control, Node, Outcome};
use arco_runtime::runner::Runner;

#[derive(Clone, Debug, PartialEq)]
struct Tick(usize);

/// Publishes until asked to stop.
struct Ticker {
    name: String,
    bus: Arc<Bus>,
    published: Arc<AtomicUsize>,
}

impl Node for Ticker {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&mut self, control: &Control) -> Result<(), Error> {
        let mut count = 0_usize;
        while !control.stop_requested() {
            self.bus.publish(&Tick(count));
            self.published.fetch_add(1, Ordering::Relaxed);
            count = count.saturating_add(1);
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
}

/// Returns immediately.
struct Immediate {
    name: String,
}

impl Node for Immediate {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&mut self, _control: &Control) -> Result<(), Error> {
        Ok(())
    }
}

/// Fails immediately.
struct Failing {
    name: String,
}

impl Node for Failing {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&mut self, _control: &Control) -> Result<(), Error> {
        Err(Error::TooFew {
            quantity: "sensor readings",
            minimum: 1,
            actual: 0,
        })
    }
}

/// Panics immediately.
struct Panicking {
    name: String,
}

impl Node for Panicking {
    fn name(&self) -> &str {
        &self.name
    }

    #[expect(
        clippy::panic,
        reason = "this node exists to panic, so the runner can be shown containing one"
    )]
    fn run(&mut self, _control: &Control) -> Result<(), Error> {
        panic!("a broken invariant");
    }
}

fn wait_until(predicate: impl Fn() -> bool) -> bool {
    for _ in 0..2000 {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

#[test]
fn a_node_runs_until_it_is_asked_to_stop() {
    let mut runner = Runner::new(64);
    let published = Arc::new(AtomicUsize::new(0));
    let ticks = runner.bus().subscribe::<Tick>();

    runner
        .spawn(Box::new(Ticker {
            name: "ticker".to_owned(),
            bus: Arc::clone(runner.bus()),
            published: Arc::clone(&published),
        }))
        .unwrap();

    assert!(wait_until(|| published.load(Ordering::Relaxed) > 3));
    let outcomes = runner.stop_all(None);

    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].1, Outcome::Completed);
    assert!(ticks.try_next().is_some(), "the node published to the bus");
}

#[test]
fn a_failing_node_reports_why() {
    // The Python runner caught every exception and discarded it, so a
    // node that crashed looked exactly like one that finished.
    let mut runner = Runner::new(8);
    runner
        .spawn(Box::new(Failing {
            name: "sensor".to_owned(),
        }))
        .unwrap();

    let outcomes = runner.stop_all(None);
    match &outcomes[0].1 {
        Outcome::Failed(message) => {
            assert!(message.contains("sensor readings"), "{message}");
        }
        other => panic!("expected a failure, got {other:?}"),
    }
    assert_eq!(runner.failures().len(), 1);
}

#[test]
fn a_panicking_node_is_contained_and_named() {
    // A panic in one node must not take the process down, and it must be
    // distinguishable from an anticipated failure.
    let mut runner = Runner::new(8);
    runner
        .spawn(Box::new(Panicking {
            name: "planner".to_owned(),
        }))
        .unwrap();

    let outcomes = runner.stop_all(None);
    match &outcomes[0].1 {
        Outcome::Panicked(message) => {
            assert!(message.contains("broken invariant"), "{message}");
        }
        other => panic!("expected a panic, got {other:?}"),
    }
    assert_eq!(outcomes[0].0, "planner");
}

#[test]
fn one_failing_node_does_not_stop_the_others() {
    let mut runner = Runner::new(64);
    let published = Arc::new(AtomicUsize::new(0));

    runner
        .spawn(Box::new(Failing {
            name: "broken".to_owned(),
        }))
        .unwrap();
    runner
        .spawn(Box::new(Ticker {
            name: "healthy".to_owned(),
            bus: Arc::clone(runner.bus()),
            published: Arc::clone(&published),
        }))
        .unwrap();

    assert!(wait_until(|| published.load(Ordering::Relaxed) > 3));
    let outcomes = runner.stop_all(None);

    assert_eq!(outcomes.len(), 2);
    assert_eq!(runner.failures().len(), 1);
    assert!(
        outcomes
            .iter()
            .any(|(name, outcome)| name == "healthy" && *outcome == Outcome::Completed),
        "{outcomes:?}"
    );
}

#[test]
fn a_node_that_returns_on_its_own_is_reported_complete() {
    let mut runner = Runner::new(8);
    runner
        .spawn(Box::new(Immediate {
            name: "once".to_owned(),
        }))
        .unwrap();

    assert!(wait_until(|| runner.running_count() == 0));
    assert_eq!(runner.outcomes()[0].1, Outcome::Completed);
    assert!(runner.failures().is_empty());
}

#[test]
fn stopping_asks_every_node_before_waiting_for_any() {
    // Stopping one at a time would let each node keep running for as long
    // as the ones before it took to shut down.
    let mut runner = Runner::new(64);
    let published = Arc::new(AtomicUsize::new(0));

    for index in 0..4 {
        runner
            .spawn(Box::new(Ticker {
                name: format!("ticker-{index}"),
                bus: Arc::clone(runner.bus()),
                published: Arc::clone(&published),
            }))
            .unwrap();
    }

    assert!(wait_until(|| published.load(Ordering::Relaxed) > 8));
    let outcomes = runner.stop_all(Some(Duration::from_secs(5)));

    assert_eq!(outcomes.len(), 4);
    for (name, outcome) in &outcomes {
        assert_eq!(*outcome, Outcome::Completed, "{name} did not stop");
    }
    assert_eq!(runner.running_count(), 0);
}

#[test]
fn a_timeout_reports_that_the_node_is_still_running() {
    // A node ignoring the stop flag cannot be forced down, and saying it
    // stopped would be worse than saying it did not.
    struct Stubborn {
        name: String,
    }
    impl Node for Stubborn {
        fn name(&self) -> &str {
            &self.name
        }
        fn run(&mut self, _control: &Control) -> Result<(), Error> {
            std::thread::sleep(Duration::from_millis(400));
            Ok(())
        }
    }

    let mut runner = Runner::new(8);
    runner
        .spawn(Box::new(Stubborn {
            name: "stubborn".to_owned(),
        }))
        .unwrap();

    let outcomes = runner.stop_all(Some(Duration::from_millis(20)));
    assert_eq!(outcomes[0].1, Outcome::Running, "{outcomes:?}");
}

#[test]
fn dropping_a_runner_asks_its_nodes_to_stop() {
    let published = Arc::new(AtomicUsize::new(0));
    let bus;
    {
        let mut runner = Runner::new(64);
        bus = Arc::clone(runner.bus());
        runner
            .spawn(Box::new(Ticker {
                name: "ticker".to_owned(),
                bus: Arc::clone(&bus),
                published: Arc::clone(&published),
            }))
            .unwrap();
        assert!(wait_until(|| published.load(Ordering::Relaxed) > 3));
    }

    let seen = published.load(Ordering::Relaxed);
    assert!(wait_until(|| {
        std::thread::sleep(Duration::from_millis(30));
        published.load(Ordering::Relaxed) == seen || published.load(Ordering::Relaxed) > seen
    }));
    // The node stops shortly after the runner goes away rather than
    // publishing into a bus nobody reads for the life of the process.
    let settled = published.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(published.load(Ordering::Relaxed), settled);
}
