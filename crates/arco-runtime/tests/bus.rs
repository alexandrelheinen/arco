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

//! The bus routes by type, bounds its queues, and never blocks a producer.

use std::sync::Arc;
use std::thread;

use arco_runtime::bus::{Bus, DEFAULT_CAPACITY};

#[derive(Clone, Debug, PartialEq)]
struct Pose(f64, f64);

#[derive(Clone, Debug, PartialEq)]
struct Plan(usize);

#[test]
fn a_frame_reaches_every_subscriber_of_its_type() {
    let bus = Bus::new(8);
    let first = bus.subscribe::<Pose>();
    let second = bus.subscribe::<Pose>();

    let report = bus.publish(&Pose(1.0, 2.0));
    assert_eq!(report.delivered, 2);
    assert_eq!(report.dropped, 0);
    assert_eq!(first.try_next(), Some(Pose(1.0, 2.0)));
    assert_eq!(second.try_next(), Some(Pose(1.0, 2.0)));
}

#[test]
fn frames_route_by_type_and_do_not_cross() {
    let bus = Bus::new(8);
    let poses = bus.subscribe::<Pose>();
    let plans = bus.subscribe::<Plan>();

    bus.publish(&Pose(1.0, 2.0));
    bus.publish(&Plan(7));

    assert_eq!(poses.try_next(), Some(Pose(1.0, 2.0)));
    assert!(poses.try_next().is_none());
    assert_eq!(plans.try_next(), Some(Plan(7)));
}

#[test]
fn publishing_with_no_subscriber_is_not_an_error() {
    let bus = Bus::new(8);
    let report = bus.publish(&Pose(0.0, 0.0));
    assert_eq!(report.delivered, 0);
    assert_eq!(report.dropped, 0);
}

#[test]
fn a_full_queue_drops_the_frame_rather_than_blocking() {
    // The property the whole design exists for: a renderer that stops
    // reading must not be able to stall a control loop.
    let bus = Bus::new(2);
    let slow = bus.subscribe::<Pose>();

    assert_eq!(bus.publish(&Pose(1.0, 0.0)).delivered, 1);
    assert_eq!(bus.publish(&Pose(2.0, 0.0)).delivered, 1);

    let report = bus.publish(&Pose(3.0, 0.0));
    assert_eq!(report.delivered, 0);
    assert_eq!(
        report.dropped, 1,
        "the third frame should have been dropped"
    );
    assert_eq!(slow.len(), 2, "the queue stayed at its bound");
}

#[test]
fn a_drop_hits_only_the_subscriber_that_is_behind() {
    let bus = Bus::new(1);
    let slow = bus.subscribe::<Pose>();
    let fast = bus.subscribe::<Pose>();

    bus.publish(&Pose(1.0, 0.0));
    let _ = fast.try_next();

    let report = bus.publish(&Pose(2.0, 0.0));
    assert_eq!(
        report.delivered, 1,
        "the caught-up subscriber still gets it"
    );
    assert_eq!(report.dropped, 1, "the slow one does not");
    assert_eq!(slow.len(), 1);
}

#[test]
fn a_zero_capacity_bus_is_unbounded() {
    // Matching queue.Queue(maxsize=0), and worth a test because the
    // number reads like "no room" rather than "no limit".
    let bus = Bus::new(0);
    let subscription = bus.subscribe::<Pose>();
    for index in 0..1000 {
        assert_eq!(bus.publish(&Pose(f64::from(index), 0.0)).dropped, 0);
    }
    assert_eq!(subscription.len(), 1000);
}

#[test]
fn draining_takes_the_newest_and_discards_the_backlog() {
    let bus = Bus::new(16);
    let subscription = bus.subscribe::<Plan>();
    for index in 0..5 {
        bus.publish(&Plan(index));
    }
    assert_eq!(subscription.drain_latest(), Some(Plan(4)));
    assert!(subscription.is_empty());
    assert!(subscription.drain_latest().is_none());
}

#[test]
fn a_late_subscriber_gets_what_follows_rather_than_what_preceded() {
    let bus = Bus::new(8);
    bus.publish(&Pose(1.0, 0.0));

    let late = bus.subscribe::<Pose>();
    assert!(late.try_next().is_none(), "history is not replayed");

    bus.publish(&Pose(2.0, 0.0));
    assert_eq!(late.try_next(), Some(Pose(2.0, 0.0)));
}

#[test]
fn dropping_a_subscription_unsubscribes_it() {
    let bus = Bus::new(8);
    let subscription = bus.subscribe::<Pose>();
    assert_eq!(bus.subscriber_count::<Pose>(), 1);

    drop(subscription);
    assert_eq!(bus.subscriber_count::<Pose>(), 0);
    assert_eq!(bus.publish(&Pose(0.0, 0.0)).delivered, 0);
}

#[test]
fn dead_subscriptions_do_not_accumulate() {
    // A long-running pipeline builds and discards renderers, and each one
    // would otherwise leave a dead reference behind forever.
    let bus = Bus::new(8);
    for _ in 0..100 {
        let subscription = bus.subscribe::<Pose>();
        bus.publish(&Pose(0.0, 0.0));
        drop(subscription);
    }
    bus.publish(&Pose(0.0, 0.0));
    assert_eq!(bus.subscriber_count::<Pose>(), 0);
}

#[test]
fn the_bus_carries_frames_across_threads() {
    let bus = Arc::new(Bus::with_default_capacity());
    assert_eq!(bus.capacity(), DEFAULT_CAPACITY);
    let subscription = bus.subscribe::<Plan>();

    let producer = {
        let bus = Arc::clone(&bus);
        thread::spawn(move || {
            for index in 0..DEFAULT_CAPACITY {
                bus.publish(&Plan(index));
            }
        })
    };
    producer.join().expect("the producer thread panicked");

    let mut received = Vec::new();
    while let Some(plan) = subscription.try_next() {
        received.push(plan.0);
    }
    assert_eq!(received, (0..DEFAULT_CAPACITY).collect::<Vec<_>>());
}

#[test]
fn concurrent_publishers_do_not_lose_frames_within_capacity() {
    let bus = Arc::new(Bus::new(0));
    let subscription = bus.subscribe::<Plan>();

    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let bus = Arc::clone(&bus);
            thread::spawn(move || {
                for index in 0..50 {
                    bus.publish(&Plan(worker * 100 + index));
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("a publisher thread panicked");
    }

    assert_eq!(subscription.len(), 200);
}
