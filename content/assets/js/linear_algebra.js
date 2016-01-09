// Copyright 2015 Charles R. Hogg III and Google Inc. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.// Utility functions for animated plots.

// This file requires jStat, but I don't know how to include it from here.  For
// now, I need to make sure to include it manually.

function Sequence(from, to, length) {
  return Array.apply(0, Array(length)).map(
      function(_, i) { return from + (to * i) / (length - 1); });
}

// This function doesn't check that M actually *has* a Cholesky decomposition
// (i.e., that it's a positive-definite symmetric matrix).  The caller is
// responsible to ensure this.
function Cholesky(M) {
  n = M.length;

  // Start out with a matrix which is all zeroes, except for the upper-left
  // element, which is trivial to compute from M.
  L = jStat(M).multiply(0);
  L[0][0] = Math.sqrt(M[0][0]);
  if (n == 1) { return L; }

  // Compute each row's values.
  for (var row = 1; row < n; ++row) {
    // This loop computes all the "pre-diagonal" elements.
    for (var col = 0; col < row; ++col) {
      // sum_to_subtract is the contribution from the elements in this new row
      // which we've already computed.
      var sum_to_subtract = 0.0;
      for (var i = 0; i < col; ++i) {
        sum_to_subtract += L[col][i] * L[row][i];
      }
      L[row][col] = (M[row][col] - sum_to_subtract) / L[col][col]
    }
    // Now, compute the element on the main diagonal.
    var sum_to_subtract = 0.0;
    for (var i = 0; i < row; ++i) {
      sum_to_subtract += L[row][i] * L[row][i];
    }
    L[row][row] = Math.sqrt(M[row][row] - sum_to_subtract);
  }

  return L;
};

// Rather an unusual function: we take the Cholesky decomposition, and we
// replace all rows but the last with a cycled version of the last row.  This is
// useful for animations.
function LoopingCholesky(M) {
  var L = Cholesky(M);
  var n = jStat.rows(M);
  for (var row = n - 2; row >= 0; --row) {
    for (var col = 1; col < n; ++col) {
      L[row][col - 1] = L[row + 1][col];
    }
    L[row][n - 1] = L[row + 1][0];
  }
  return L;
};

// Creates a deep copy of the given jStat matrix.
//
// Naively assumes that M is well formed (e.g., every row has the same number of
// columns).
//
// Args:
//    M:  A matrix (assumed equivalent to the output of jStat.create()).
//
// Returns:
//    An array-of-arrays equivalent to the given matrix.
function DeepCopy(M) {
  return jStat.create(M.length, M[0].length, function(i, j) {
    return M[i][j];
  });
}

// Matrix for an n-D Givens rotation about columns i and j by angle theta.
function GivensRotationMatrix(n, i, j, theta) {
  var c = Math.cos(theta);
  var s = Math.sin(theta);
  var g = jStat.identity(n);
  g[i][i] = g[j][j] = c;
  g[i][j] = -s;
  g[j][i] = s;
  return g;
}

// The index of the super-diagonal element with maximal magnitude.
function IndexOfMaxSuperDiagonalInRow(mat, row) {
  var max_value = undefined;
  var i_best = -1;
  for (var i = row + 1; i < mat[row].length; i++) {
    var value = Math.abs(mat[row][i]);
    if (typeof max_value === "undefined" || value > max_value) {
      max_value = value;
      i_best = i;
    }
  }
  return i_best;
}

// The "pivot row" is a row which contains a highest off-diagonal element.
//
// We assume that the i_columns array gives the index of the highest
// off-diagonal element for that row.
function PivotRow(mat, i_columns) {
  var values = i_columns.map(function(v, i) {
    return Math.abs(mat[i][v]);
  });
  return values.reduce(function(i_max, v, i, a) {
    return v > a[i_max] ? i : i_max;
  }, 0);
}

// Return the eigenvalues and eigenvectors of the matrix.
//
// This function will assume that these exist (and are real, and are
// sufficiently well-conditioned, etc.) for the given matrix.  If it returns an
// answer when these assumptions are false (which it might!), such an answer
// would be meaningless.
//
// Args:
//    M:  A matrix (structured equivalently to the output of jStat.create()).
//
// Returns:
//    {
//      vectors:  A matrix whose rows are the eigenvectors of M.
//      values:  An array giving the eigenvalues which correspond to the
//        eigenvectors.
//    }
function Eigen(M) {
  // Use the jacobi algorithm [1], since it is more accurate than the QR
  // algorithm [2].
  //
  // [1] https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm#Algorithm
  // [2] "Jacobi’s Method is More Accurate than QR". James Demmel and Krešimir
  //     Veselić, SIAM Journal on Matrix Analysis and Applications 1992 13:4,
  //     1204-1245 http://dx.doi.org/10.1137/0613074

  // Translate logical coordinates into actual coordinates.
  //
  // If a symmetric matrix is stored as upper-triangular only for efficiency,
  // then the values from the lower triangular region will be nonsense, and we
  // need to be careful to avoid them.
  function Coords(row, col, upper_only) {
    return (upper_only && row > col) ? [col, row] : [row, col];
  }

  // Rotate columns i and j by angle theta.
  //
  // If upper_only is true, we will only read and write from the
  // upper-triangular portion of the matrix, and we will skip elements in the
  // ith and jth row or column, since those need to be handled specially.
  //
  // Actually, having this one function handle both cases is brittle and makes
  // the code harder to read and understand.  I should refactor this. :P
  function AdHocRotation(matrix, i, j, theta, upper_only) {
    var c = Math.cos(theta);
    var s = Math.sin(theta);
    for (var row = 0; row < matrix.length; ++row) {
      if (upper_only && (row == i || row == j)) {
        // We need to handle the ii, ij/ji, and jj elements specially.
        continue;
      }
      var i_coords = Coords(row, i, upper_only);
      var j_coords = Coords(row, j, upper_only);
      var m_i = matrix[i_coords[0]][i_coords[1]];
      var m_j = matrix[j_coords[0]][j_coords[1]];
      matrix[i_coords[0]][i_coords[1]] = c * m_i - s * m_j;
      matrix[j_coords[0]][j_coords[1]] = s * m_i + c * m_j;
    }
  }

  // A copy of the matrix which we can modify without clobbering M.
  var matrix = DeepCopy(M);
  // The dimension of the (assumed-to-be-square) matrix.
  var n = matrix.length;
  // The matrix which will eventually converge to the eigenvectors of M.
  var eigenvectors = jStat.identity(n);
  // The number of rows we recently changed by a significant amount.
  var num_rows_changed = n;
  // The rows we recently changed by a significant amount.
  var rows_changed = matrix.map(function() { return true; });
  // The index, for each row, of its biggest super-diagonal element.  (We remove
  // the final element since the last row has no super-diagonal elements.)
  var max_index = matrix.map(function(v, i, a) {
    return IndexOfMaxSuperDiagonalInRow(a, i);
  });
  max_index.pop()
  // The threshold for the highest nonzero element to be considered "close
  // enough" to zero.
  var pivot_threshold = 1e-16;
  // The number of iterations we have taken.
  var iter = 0;

  // Iterate until our pivot is under the threshold n times consecutively.
  var num_good_in_a_row = n;
  var countdown = num_good_in_a_row;
  while (countdown > 0) {
    iter++;
    // Find the location of the pivot.
    var pivot_row = PivotRow(matrix, max_index);
    var pivot_column = max_index[pivot_row];
    // Check the end condition.
    countdown = (Math.abs(matrix[pivot_row][pivot_column]) < pivot_threshold) ?
      (countdown - 1) : num_good_in_a_row;
    // Compute the Givens rotation angle and matrix.
    var theta = 0.5 * Math.atan2(
        2 * matrix[pivot_row][pivot_column],
        matrix[pivot_column][pivot_column] - matrix[pivot_row][pivot_row]);
    // Apply the Givens matrix to the input matrix and the eigenvector matrix.
    AdHocRotation(matrix, pivot_row, pivot_column, theta,
                  /* upper_only = */ true);
    // Handle the ii, ij/ji, and jj elements specially.
    delta_i_j = (
        Math.pow(Math.sin(theta), 2) * (
          matrix[pivot_row][pivot_row] - matrix[pivot_column][pivot_column]) +
        Math.sin(2 * theta) * matrix[pivot_row][pivot_column]);
    matrix[pivot_row][pivot_row] -= delta_i_j;
    matrix[pivot_column][pivot_column] += delta_i_j;
    matrix[pivot_row][pivot_column] = matrix[pivot_column][pivot_row] = 0;
    // Update the max_index array.
    max_index[pivot_row] = IndexOfMaxSuperDiagonalInRow(matrix, pivot_row);
    if (pivot_column < matrix.length - 1) {
      max_index[pivot_column] = IndexOfMaxSuperDiagonalInRow(
          matrix, pivot_column);
    }
    // Update the eigenvectors too.
    AdHocRotation(eigenvectors, pivot_row, pivot_column, theta,
                  /* upper_only = */ false);
  }

  return {
    vectors: jStat.transpose(eigenvectors),
    values: jStat.transpose(jStat.diag(matrix)),
  }
}

// A diagonal matrix whose diagonal elements are as given.
function DiagonalMatrix(values) {
  var n = values.length;
  var matrix = jStat.identity(n);
  for (var i = 0; i < n; i++) {
    matrix[i][i] = values[i];
  }
  return matrix;
}

// A symmetric square root of the covariance matrix K.
function SymmetricSquareRoot(K) {
  var eigen = Eigen(K);
  return jStat.multiply(
      jStat.transpose(eigen.vectors),
      jStat.multiply(
        DiagonalMatrix(eigen.values.map(function(v) { return Math.sqrt(v); })),
        eigen.vectors));
}
