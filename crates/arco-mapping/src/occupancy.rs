//! A k-d tree over obstacle points, and the queries planners make of it.

use arco_core::Error;
use arco_core::geometry::{require_dimension, require_finite};
use arco_core::protocols::{NearestObstacle, Occupancy, SegmentChecker};

/// One node of the tree: a split point and the axis it splits on.
#[derive(Debug, Clone)]
struct Node {
    /// Index into the point store.
    point: usize,
    /// Axis this node splits on.
    axis: usize,
    /// Subtree of points below the split.
    left: Option<Box<Node>>,
    /// Subtree of points at or above the split.
    right: Option<Box<Node>>,
}

/// An obstacle field described by a set of points and a clearance radius.
///
/// A query point is occupied when it lies within `clearance` of any
/// obstacle point, so the clearance is what turns a point cloud into a
/// region.
///
/// The tree is built once at construction and never mutated, which is
/// what lets [`KdTreeOccupancy::content_hash`] be computed once and lets a
/// planner state which field it planned against.
#[derive(Debug, Clone)]
pub struct KdTreeOccupancy {
    points: Vec<Vec<f64>>,
    dimension: usize,
    clearance: f64,
    root: Option<Node>,
    content_hash: u64,
}

impl KdTreeOccupancy {
    /// Builds an occupancy from obstacle points.
    ///
    /// # Arguments
    ///
    /// * `points` - Obstacle points, all of the same dimension.
    /// * `clearance` - Radius around each point that counts as occupied,
    ///   meters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when `points` is empty,
    /// [`Error::DimensionMismatch`] when the points disagree,
    /// [`Error::NotFinite`] when one carries a NaN, and
    /// [`Error::OutOfRange`] when `clearance` is negative or not finite.
    pub fn new(points: &[Vec<f64>], clearance: f64) -> Result<Self, Error> {
        let Some(first) = points.first() else {
            return Err(Error::TooFew {
                quantity: "obstacle points",
                minimum: 1,
                actual: 0,
            });
        };
        let dimension = first.len();
        if dimension == 0 {
            return Err(Error::TooFew {
                quantity: "obstacle point coordinates",
                minimum: 1,
                actual: 0,
            });
        }
        if !(clearance.is_finite() && clearance >= 0.0) {
            return Err(Error::OutOfRange {
                quantity: "clearance",
                value: clearance,
                bound: "[0, inf)",
            });
        }
        for point in points {
            require_dimension("obstacle point", point, dimension)?;
            require_finite("obstacle point", point)?;
        }

        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = build(points, &mut indices, 0, dimension);
        let content_hash = hash_points(points, clearance);

        Ok(Self {
            points: points.to_vec(),
            dimension,
            clearance,
            root,
            content_hash,
        })
    }

    /// The obstacle points, in the order they were given.
    #[must_use]
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// The clearance radius, meters.
    #[must_use]
    pub const fn clearance(&self) -> f64 {
        self.clearance
    }

    /// A hash of the points and the clearance.
    ///
    /// `FR-INV-12`, the occupancy counterpart of the grid's content hash.
    #[must_use]
    pub const fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// The distance from `point` to every obstacle's surface.
    ///
    /// # Errors
    ///
    /// As [`KdTreeOccupancy::nearest_obstacle`].
    pub fn query_distances(&self, points: &[Vec<f64>]) -> Result<Vec<f64>, Error> {
        points
            .iter()
            .map(|point| Ok(self.nearest_obstacle(point)?.distance))
            .collect()
    }
}

impl Occupancy for KdTreeOccupancy {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn clearance(&self) -> f64 {
        self.clearance
    }

    fn nearest_obstacle(&self, point: &[f64]) -> Result<NearestObstacle, Error> {
        require_dimension("query point", point, self.dimension)?;
        require_finite("query point", point)?;

        let mut best = (f64::INFINITY, 0_usize);
        search(self.root.as_ref(), &self.points, point, &mut best);

        let (squared, index) = best;
        let nearest = self.points.get(index).cloned().unwrap_or_default();
        Ok(NearestObstacle {
            // Distance to the obstacle surface rather than to its center,
            // which is what a planner asking for clearance means.
            distance: squared.sqrt() - self.clearance,
            point: nearest,
        })
    }

    fn is_occupied(&self, point: &[f64]) -> Result<bool, Error> {
        Ok(self.nearest_obstacle(point)?.distance <= 0.0)
    }

    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        require_dimension("segment start", from, self.dimension)?;
        require_dimension("segment end", to, self.dimension)?;
        require_finite("segment start", from)?;
        require_finite("segment end", to)?;

        // The capsule of radius `clearance` around the segment is what
        // has to stay empty, and its bounding box is what prunes the
        // descent. A subtree lying wholly on the far side of a split
        // plane from that box cannot hold a point inside it.
        let bounds: Vec<(f64, f64)> = from
            .iter()
            .zip(to)
            .map(|(start, end)| {
                (
                    start.min(*end) - self.clearance,
                    start.max(*end) + self.clearance,
                )
            })
            .collect();

        Ok(!intersects(
            self.root.as_ref(),
            &self.points,
            from,
            to,
            &bounds,
            self.clearance,
        ))
    }
}

impl SegmentChecker for KdTreeOccupancy {
    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        Occupancy::is_segment_free(self, from, to)
    }
}

impl KdTreeOccupancy {
    /// Whether a segment is free, sampling it `sample_count` times.
    ///
    /// Sampling is what the Python implementation did and what the
    /// planners were tuned against. It is not exact: a thin obstacle
    /// between two samples is missed, which is why a returned path is
    /// re-checked under `FR-INV-01` at a stated resolution rather than
    /// declared safe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the endpoints disagree.
    pub fn is_segment_free_with(
        &self,
        from: &[f64],
        to: &[f64],
        sample_count: usize,
    ) -> Result<bool, Error> {
        require_dimension("segment end", to, from.len())?;
        let count = sample_count.max(2);
        let divisor = f64::from(u32::try_from(count.saturating_sub(1)).unwrap_or(1));

        for step in 0..count {
            let ratio = f64::from(u32::try_from(step).unwrap_or(0)) / divisor;
            let sample: Vec<f64> = from
                .iter()
                .zip(to)
                .map(|(start, end)| start + (end - start) * ratio)
                .collect();
            if self.is_occupied(&sample)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Builds a balanced subtree over `indices`, splitting on a rotating axis.
fn build(
    points: &[Vec<f64>],
    indices: &mut [usize],
    depth: usize,
    dimension: usize,
) -> Option<Node> {
    if indices.is_empty() {
        return None;
    }
    let axis = depth.checked_rem(dimension).unwrap_or(0);

    // A total order, so an obstacle set containing equal coordinates
    // builds the same tree on every run.
    indices.sort_by(|&left, &right| {
        coordinate(points, left, axis).total_cmp(&coordinate(points, right, axis))
    });
    let middle = indices.len().checked_div(2).unwrap_or(0);
    let (lower, rest) = indices.split_at_mut(middle);
    let (pivot, upper) = rest.split_first_mut()?;

    Some(Node {
        point: *pivot,
        axis,
        left: build(points, lower, depth.saturating_add(1), dimension).map(Box::new),
        right: build(points, upper, depth.saturating_add(1), dimension).map(Box::new),
    })
}

/// Descends the tree, keeping the closest point found so far.
fn search(node: Option<&Node>, points: &[Vec<f64>], query: &[f64], best: &mut (f64, usize)) {
    let Some(node) = node else { return };

    let squared = squared_distance(points, node.point, query);
    if squared < best.0 {
        *best = (squared, node.point);
    }

    let offset =
        query.get(node.axis).copied().unwrap_or(0.0) - coordinate(points, node.point, node.axis);
    let (near, far) = if offset < 0.0 {
        (node.left.as_deref(), node.right.as_deref())
    } else {
        (node.right.as_deref(), node.left.as_deref())
    };

    search(near, points, query, best);
    // The far side can only hold something closer if the splitting plane
    // itself is closer than the best distance found so far.
    if offset * offset < best.0 {
        search(far, points, query, best);
    }
}

fn coordinate(points: &[Vec<f64>], index: usize, axis: usize) -> f64 {
    points
        .get(index)
        .and_then(|point| point.get(axis))
        .copied()
        .unwrap_or(f64::INFINITY)
}

fn squared_distance(points: &[Vec<f64>], index: usize, query: &[f64]) -> f64 {
    points.get(index).map_or(f64::INFINITY, |point| {
        point
            .iter()
            .zip(query)
            .map(|(a, b)| {
                let difference = a - b;
                difference * difference
            })
            .sum()
    })
}

/// Hashes the obstacle set and its clearance, order-independently.
fn hash_points(points: &[Vec<f64>], clearance: f64) -> u64 {
    let mut hash = mix(clearance.to_bits());
    for point in points {
        let mut term = 0x9e37_79b9_7f4a_7c15_u64;
        for value in point {
            term = mix(term ^ value.to_bits());
        }
        hash ^= term;
    }
    hash
}

const fn mix(value: u64) -> u64 {
    let mut mixed = value;
    mixed ^= mixed >> 33;
    mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
    mixed ^= mixed >> 33;
    mixed = mixed.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    mixed ^ (mixed >> 33)
}

/// Whether any obstacle lies within `clearance` of the segment.
///
/// Descends the tree, keeping a subtree only while the query's bounding
/// box reaches across the split plane into it. The walk is over the tree
/// rather than over the points, so a sparse field costs a handful of
/// distance evaluations rather than one per obstacle.
fn intersects(
    node: Option<&Node>,
    points: &[Vec<f64>],
    from: &[f64],
    to: &[f64],
    bounds: &[(f64, f64)],
    clearance: f64,
) -> bool {
    let Some(node) = node else { return false };
    let Some(point) = points.get(node.point) else {
        return false;
    };

    if squared_distance_to_segment(point, from, to) <= clearance * clearance {
        return true;
    }

    let split = point.get(node.axis).copied().unwrap_or_default();
    let (low, high) = bounds
        .get(node.axis)
        .copied()
        .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));

    // Every point left of the split sits at or below it, so the whole
    // subtree is out of reach once the split is below the box.
    if split >= low && intersects(node.left.as_deref(), points, from, to, bounds, clearance) {
        return true;
    }
    if split <= high && intersects(node.right.as_deref(), points, from, to, bounds, clearance) {
        return true;
    }
    false
}

/// The squared distance from `point` to the segment `from` to `to`.
fn squared_distance_to_segment(point: &[f64], from: &[f64], to: &[f64]) -> f64 {
    let mut span_squared = 0.0_f64;
    let mut projection = 0.0_f64;
    for ((start, end), coordinate) in from.iter().zip(to).zip(point) {
        let span = end - start;
        span_squared += span * span;
        projection += (coordinate - start) * span;
    }
    // A degenerate segment is a point, and the clamp below would divide
    // by zero rather than say so.
    let ratio = if span_squared > 0.0 {
        (projection / span_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut total = 0.0_f64;
    for ((start, end), coordinate) in from.iter().zip(to).zip(point) {
        let offset = (end - start).mul_add(ratio, start - coordinate);
        total += offset * offset;
    }
    total
}
