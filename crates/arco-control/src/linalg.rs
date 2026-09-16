//! The small linear algebra the actuator allocation needs, and no more.
//!
//! Force allocation inverts a three by two-N grasp matrix, which numpy did
//! with a singular value decomposition. The identity `A+ = A^T (A A^T)+`
//! holds for any real matrix, and `A A^T` here is three by three, so the
//! decomposition needed is of a symmetric three by three rather than of
//! the full matrix. That is a cyclic Jacobi rotation, about forty lines,
//! which is why this crate has no linear algebra dependency.

use arco_core::Error;

/// A symmetric three by three matrix, stored by its upper triangle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Symmetric3 {
    entries: [[f64; 3]; 3],
}

impl Symmetric3 {
    /// The zero matrix.
    pub(crate) const fn zero() -> Self {
        Self {
            entries: [[0.0; 3]; 3],
        }
    }

    /// Adds `value` at `(row, column)` and at its mirror.
    ///
    /// Only the two together keep the matrix symmetric, which every
    /// routine below assumes.
    pub(crate) fn add(&mut self, row: usize, column: usize, value: f64) {
        if let Some(entry) = self.entries.get_mut(row).and_then(|r| r.get_mut(column)) {
            *entry += value;
        }
        if row != column
            && let Some(entry) = self.entries.get_mut(column).and_then(|r| r.get_mut(row))
        {
            *entry += value;
        }
    }

    /// The entry at `(row, column)`, or zero when out of range.
    fn get(&self, row: usize, column: usize) -> f64 {
        self.entries
            .get(row)
            .and_then(|r| r.get(column))
            .copied()
            .unwrap_or_default()
    }

    /// Sets the entry at `(row, column)`.
    fn set(&mut self, row: usize, column: usize, value: f64) {
        if let Some(entry) = self.entries.get_mut(row).and_then(|r| r.get_mut(column)) {
            *entry = value;
        }
    }

    /// Solves `self * x = right` in the least-squares, minimum-norm sense.
    ///
    /// The pseudo-inverse rather than the inverse, because a grasp matrix
    /// loses rank whenever the actuators cannot produce a wrench in some
    /// direction, and that is a configuration a caller is allowed to ask
    /// about rather than a programming error. A singular direction
    /// contributes nothing instead of producing an infinity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when an entry is not a real number,
    /// since the rotation below would otherwise spread the NaN over every
    /// output.
    pub(crate) fn solve_pseudo(&self, right: [f64; 3]) -> Result<[f64; 3], Error> {
        for row in 0..3 {
            for column in 0..3 {
                let value = self.get(row, column);
                if !value.is_finite() {
                    return Err(Error::NotFinite {
                        quantity: "grasp matrix entry",
                        value,
                    });
                }
            }
        }
        for value in right {
            if !value.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "desired wrench",
                    value,
                });
            }
        }

        let (values, vectors) = self.eigen();
        let largest = values
            .iter()
            .fold(0.0_f64, |best, value| best.max(value.abs()));
        // The usual pseudo-inverse cutoff: anything this far below the
        // largest singular value is indistinguishable from zero at double
        // precision, and dividing by it amplifies rounding into the answer.
        let cutoff = largest * 3.0 * f64::EPSILON;

        let mut solution = [0.0_f64; 3];
        for (index, &value) in values.iter().enumerate() {
            if value.abs() <= cutoff {
                continue;
            }
            // The component of `right` along this eigenvector, scaled by
            // the reciprocal eigenvalue and put back.
            let mut projection = 0.0;
            for row in 0..3 {
                projection += column_entry(&vectors, row, index) * get(&right, row);
            }
            let scaled = projection / value;
            for (row, slot) in solution.iter_mut().enumerate() {
                *slot += scaled * column_entry(&vectors, row, index);
            }
        }
        Ok(solution)
    }

    /// Eigenvalues and eigenvectors, by cyclic Jacobi rotation.
    ///
    /// Symmetric matrices have a full set of real eigenvalues and an
    /// orthogonal eigenbasis, which is what makes the rotation converge
    /// and what makes the reconstruction above exact. The sweep count is
    /// fixed rather than "until converged", so the routine has a stated
    /// bound; a three by three converges in far fewer.
    fn eigen(&self) -> ([f64; 3], [[f64; 3]; 3]) {
        let mut working = *self;
        let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        for _ in 0..20 {
            let off_diagonal =
                working.get(0, 1).abs() + working.get(0, 2).abs() + working.get(1, 2).abs();
            if off_diagonal <= f64::EPSILON {
                break;
            }
            for (row, column) in [(0_usize, 1_usize), (0, 2), (1, 2)] {
                working.rotate(&mut vectors, row, column);
            }
        }

        (
            [working.get(0, 0), working.get(1, 1), working.get(2, 2)],
            vectors,
        )
    }

    /// Zeroes the entry at `(row, column)` with one Jacobi rotation.
    fn rotate(&mut self, vectors: &mut [[f64; 3]; 3], row: usize, column: usize) {
        let off = self.get(row, column);
        if off.abs() <= f64::EPSILON {
            return;
        }
        let diagonal_difference = self.get(column, column) - self.get(row, row);
        // The rotation angle that annihilates the off-diagonal entry,
        // written through the half-angle so it stays accurate when the
        // two diagonal entries are close.
        let theta = diagonal_difference / (2.0 * off);
        let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
        let tangent = sign / (theta.abs() + theta.mul_add(theta, 1.0).sqrt());
        let cosine = 1.0 / tangent.mul_add(tangent, 1.0).sqrt();
        let sine = tangent * cosine;

        let (row_value, column_value) = (self.get(row, row), self.get(column, column));
        self.set(row, row, tangent.mul_add(-off, row_value));
        self.set(column, column, tangent.mul_add(off, column_value));
        self.set(row, column, 0.0);
        self.set(column, row, 0.0);

        for other in 0..3 {
            if other == row || other == column {
                continue;
            }
            let left = self.get(other, row);
            let right = self.get(other, column);
            self.set(other, row, cosine.mul_add(left, -(sine * right)));
            self.set(row, other, cosine.mul_add(left, -(sine * right)));
            self.set(other, column, sine.mul_add(left, cosine * right));
            self.set(column, other, sine.mul_add(left, cosine * right));
        }

        for entry in vectors.iter_mut() {
            let left = entry.get(row).copied().unwrap_or_default();
            let right = entry.get(column).copied().unwrap_or_default();
            if let Some(slot) = entry.get_mut(row) {
                *slot = cosine.mul_add(left, -(sine * right));
            }
            if let Some(slot) = entry.get_mut(column) {
                *slot = sine.mul_add(left, cosine * right);
            }
        }
    }
}

/// The `row`th entry of the `index`th eigenvector.
fn column_entry(vectors: &[[f64; 3]; 3], row: usize, index: usize) -> f64 {
    vectors
        .get(row)
        .and_then(|entries| entries.get(index))
        .copied()
        .unwrap_or_default()
}

/// A triple entry by index, zero when out of range.
fn get(values: &[f64; 3], index: usize) -> f64 {
    values.get(index).copied().unwrap_or_default()
}
