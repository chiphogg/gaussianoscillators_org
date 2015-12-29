// Copyright 2015 Google Inc. All Rights Reserved.
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
function DeepCopyMatrixOnly(M) {
  var matrix = jStat.zeros(M[0].length, M[0][0].length);
  for (var i = 0; i < matrix.length; i++) {
    for (var j = 0; j < matrix[i].length; j++) {
      matrix[i][j] = M[0][i][j];
    }
  }
  return matrix;
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

  //////////////////////////////////////////////////////////////////////////////
  // Helper functions.

  // The index of the off-diagonal element with maximal magnitude.
  function IndexOfMaxOffDiagonalInRow(mat, row) {
    var max_value = undefined;
    var i_best = -1;
    for (var i = 0; i < mat[row].length; i++) {
      if (i == row) { continue; }
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

  // Alter M in-place with a Givens rotation about columns i and j by angle
  // theta.
  function ApplyGivensRotation(M, i, j, theta) {
    var c = Math.cos(theta);
    var s = Math.sin(theta);
    console.log('c: ' + c + '; s: ' + s);

    // Update the diagonal elements (ii and jj).  (We'll need to save some
    // information about their old values first!)
    var new_cross_diagonal = (
        (c * c - s * s) * M[i][j] +
        s * c * (M[i][i] - M[j][j]));
    M[i][i] = c * c * M[i][i] - 2 * s * c * M[i][j] + s * s * M[j][j];
    M[j][j] = s * s * M[i][i] + 2 * s * c * M[i][j] + c * c * M[j][j];
    console.log(M[i][i]);

    // Update the cross-diagonal elements (ij and ji).
    M[i][j] = M[j][i] = new_cross_diagonal;

    // Update the off-diagonal elements (those with only one of i or j).
    for (var k = 0; k < M.length; k++) {
      if (k == i || k == j) {
        continue;
      }
      var new_i = c * M[i][k] - s * M[j][k];
      var new_j = s * M[i][k] + c * M[j][k];
      M[i][k] = M[k][i] = new_i;
      M[j][k] = M[k][j] = new_j;
    }
  }

  // Zero out the (i, j) and (j, i) entries, logging them for posterity (so we
  // can make sure they were already small).
  function ZeroOutEntries(M, i, j) {
    console.log('i: ' + i + '; j: ' + j +
        '; (i, j): ' + M[i][j] + '; (j, i): ' + M[j][i]);
    M[i][j] = M[j][i] = 0;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Variable initializations.

  // A copy of the matrix which we can modify without clobbering M.
  var matrix = DeepCopyMatrixOnly(M);
  // The dimension of the (assumed-to-be-square) matrix.
  var n = matrix.length;
  // The matrix which will eventually converge to the eigenvectors of M.
  var eigenvectors = jStat.identity(n);
  // The vector which will eventually converge to the eigenvalues of M.
  var eigenvalues = jStat.transpose(jStat.diag(matrix));
  // The number of rows we recently changed by a significant amount.
  var num_rows_changed = n;
  // The rows we recently changed by a significant amount.
  var rows_changed = matrix.map(function() { return true; });
  // The index, for each row, of its biggest off-diagonal element.
  var max_index = matrix.map(function(v, i, a) {
    return IndexOfMaxOffDiagonalInRow(a, i);
  });

  //////////////////////////////////////////////////////////////////////////////
  // Logic.

  while (num_rows_changed > 0) {
    // Find the location of the pivot.
    var pivot_row = PivotRow(matrix, max_index);
    var pivot_column = max_index[pivot_row];
    // Compute the Givens rotation angle.
    var theta = 0.5 * Math.atan2(2 * matrix[i][j], matrix[j][j] - matrix[i][i]);
  }

  return {
    vectors: eigenvectors,
    values: eigenvalues,
  }
}

