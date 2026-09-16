//! Shortening a raw path to the fewest waypoints that still connect.

use std::collections::VecDeque;

use arco_core::Error;
use arco_core::protocols::{Occupancy, Pruner};

use super::policy::SegmentPolicy;

/// Reduces a planner's raw path to the fewest waypoints that connect.
///
/// A sampling planner returns a path made of steering steps, which is far
/// more waypoints than the geometry needs. Dropping the redundant ones is
/// worth doing before a trajectory optimizer sees the path, because the
/// optimizer's problem size is the waypoint count.
///
/// The reduction is a breadth-first search over path indices rather than
/// a greedy forward scan, and the difference matters: reachability along
/// a path is not monotone, since an obstacle can block a short shortcut
/// while a longer one goes around it. A greedy scan stops at the first
/// blocked jump and keeps waypoints it did not need. The search returns
/// the fewest waypoints that exist.
#[derive(Debug)]
pub struct TrajectoryPruner<O> {
    segments: SegmentPolicy<O>,
}

impl<O: Occupancy> TrajectoryPruner<O> {
    /// Builds a pruner that shortcuts through `segments`.
    #[must_use]
    pub const fn new(segments: SegmentPolicy<O>) -> Self {
        Self { segments }
    }

    /// The fewest waypoints of `path` that still connect end to end.
    ///
    /// `FR-INV-03`. The result is a subsequence of the input keeping its
    /// first and last state, so under any metric obeying the triangle
    /// inequality it is never longer than the input, and every edge it
    /// keeps was accepted by the segment policy. A custom cost that is
    /// not a metric can break the first of those; the second holds
    /// regardless.
    ///
    /// # Errors
    ///
    /// Propagates whatever the segment policy returns.
    pub fn shortest_subsequence(&self, path: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error> {
        let count = path.len();
        if count <= 2 {
            return Ok(path.to_vec());
        }
        let goal = count.saturating_sub(1);

        // `reached_from[i]` is the index the search first arrived at `i`
        // from, and doubles as the visited marker. The root points at
        // itself, which the trace back stops on.
        let mut reached_from: Vec<Option<usize>> = vec![None; count];
        if let Some(slot) = reached_from.get_mut(0) {
            *slot = Some(0);
        }
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(count);
        queue.push_back(0);

        while let Some(current) = queue.pop_front() {
            if current == goal {
                break;
            }
            let Some(from) = path.get(current) else {
                continue;
            };
            for next in current.saturating_add(1)..count {
                if reached_from.get(next).copied().flatten().is_some() {
                    continue;
                }
                let Some(to) = path.get(next) else { continue };
                if !self.segments.is_segment_free(from, to)? {
                    continue;
                }
                if let Some(slot) = reached_from.get_mut(next) {
                    *slot = Some(current);
                }
                queue.push_back(next);
                if next == goal {
                    break;
                }
            }
        }

        Ok(trace(path, &reached_from, goal))
    }
}

impl<O: Occupancy> Pruner for TrajectoryPruner<O> {
    fn prune(&self, path: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error> {
        self.shortest_subsequence(path)
    }
}

/// Walks the search back from `goal`, or returns `path` when it cannot.
///
/// Failing to reach the goal means two consecutive waypoints of the input
/// were not connectable, which says the input was already invalid. The
/// honest answer then is the input unchanged, since a pruner that drops
/// waypoints from a broken path makes it harder to see what broke.
fn trace(path: &[Vec<f64>], reached_from: &[Option<usize>], goal: usize) -> Vec<Vec<f64>> {
    let mut indices = Vec::new();
    let mut index = goal;
    for _ in 0..=path.len() {
        if index == 0 {
            break;
        }
        indices.push(index);
        let Some(Some(previous)) = reached_from.get(index).copied() else {
            return path.to_vec();
        };
        index = previous;
    }
    if index != 0 {
        return path.to_vec();
    }
    indices.push(0);
    indices.reverse();
    indices
        .into_iter()
        .filter_map(|kept| path.get(kept).cloned())
        .collect()
}
