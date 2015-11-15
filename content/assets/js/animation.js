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

function standardNormal() {
  return jStat.normal.sample(0, 1);
}

function newStandardNormalsForRow(matrix, row) {
  for (var i = 0; i < matrix[row].length; ++i) {
    matrix[row][i] = standardNormal();
  }
}

//------------------------------------------------------------------------------
// Gaussian oscillators.

// A Gaussian oscillator with delta-distribution covariance.  (In other words:
// every timestep is independent.)
function independentOscillator(n) {
  var noise = jStat.create(1, n, standardNormal);

  return {
    advance: function() {
      newStandardNormalsForRow(noise, 0);
    },
    currentNoise: function() {
      return noise[0];
    },
    toString: function() {
      var output = '[';
      var comma = '';
      var noise = this.currentNoise();
      for (var i = 0; i < n; ++i) {
        output += comma + noise[i].toFixed(3);
        comma = ', ';
      }
      return output + ']';
    }
  };
}

// An oscillator which interpolates (trigonometrically) between independent
// timesteps.
function interpolatingOscillator(n, n_t) {
  // Two independent n-dimensional normal samples (we will interpolate between
  // them).
  var noise = jStat.create(2, n, standardNormal);
  // The index of the most recent normal draw.
  var i_prev = 0;
  // A convenience variable to hold the current interpolated noise.
  var cachedNoise = jStat(noise[i_prev]).multiply(1)[0];
  // The index of the frame we are interpolating.
  var i = 0;

  function incrementCounter() {
    i += 1;

    // If we've reached the next independent sample: reset the counter,
    // generate new data, and mark the *other* independent sample as the
    // "most recent".
    if (i == n_t) {
      i = 0;
      newStandardNormalsForRow(noise, i_prev);
      i_prev = 1 - i_prev;
    }
  }

  function storeInterpolatedNoise() {
    var angle = (i / n_t) * (Math.PI / 2);
    var old_factor = Math.cos(angle);
    var new_factor = Math.sin(angle);
    for (var j = 0; j < cachedNoise.length; ++j) {
      cachedNoise[j] = (old_factor * noise[i_prev][j] +
                        new_factor * noise[1 - i_prev][j]); 
    }
  }

  return Object.assign(
      Object.create(independentOscillator(n)),
      {
        advance: function() {
          incrementCounter();
          storeInterpolatedNoise();
        },
        currentNoise: function() {
          return cachedNoise;
        }

      });
}

// Create a covariance matrix with compact support for a given number of equally
// spaced points.
function CompactSupportCovarianceMatrix(N) {
  return jStat.create(N, N, function(i, j) {
    var dt = Math.abs(i - j) / N;
    return (Math.pow(1 - dt, 6)
            * ((12.8 * dt * dt * dt) + (13.8 * dt * dt) + (6 * dt) + 1));
  });
}

// A CSC (Compact Support Covariance) Oscillator (Hogg 2015).
function compactSupportCovarianceOscillator(n, n_t) {
  // A matrix of independent normal samples with n_t rows.
  // At every stage, we will replace the oldest row with brand new samples.
  var random_matrix = jStat.create(n_t, n, standardNormal);
  // A matrix where each row is equivalent to the previous, but shifted by one.
  var L_t = LoopingCholesky(CompactSupportCovarianceMatrix(n_t));
  // The row of L_t which holds the vector to use.
  var i = 0;
  // A convenience variable to hold the current interpolated noise.
  var cachedNoise = interpolatedNoise();

  function interpolatedNoise() {
    return jStat(L_t[i]).multiply(random_matrix)[0];
  }

  return Object.assign(
      Object.create(independentOscillator(n)),
      {
        advance: function() {
          newStandardNormalsForRow(random_matrix, i);
          i = (i + 1) % n_t;
          cachedNoise = interpolatedNoise();
        },
        currentNoise: function() {
          return cachedNoise;
        }
      });
}

// A helper for looping Gaussian Oscillators, which store all their frames in
// memory.  This simply loops through a matrix.
function looper(matrix) {
  // The index of the current row.
  var i = 0;

  return {
    advance: function() {
      i = (i + 1) % matrix.length;
    },
    current: function() {
      return matrix[i];
    }
  };
}

function greatCircleMatrix(n, n_t) {
  // Normalize a vector in-place.
  function normalize(v) {
    var factor = 1 / Math.sqrt(jStat.dot(v, v));
    for (var i = 0; i < v.length; ++i) {
      v[i] *= factor;
    }
  }
  // An n-dimensional unit vector with a random direction.
  function randomUnitVector(n) {
    var v = jStat.create(1, n, standardNormal)[0];
    normalize(v);
    return v;
  }

  // Reserve space for the matrix.
  var mat = jStat.zeros(n_t, n);

  // Generate the first draw from the normal; record its magnitude, and
  // normalize it.
  var first = jStat.create(1, n, standardNormal)[0];
  var magnitude = Math.sqrt(jStat.dot(first, first));
  normalize(first);

  // Generate the second draw; orthogonalize it, and normalize it.
  var second = randomUnitVector(n);
  var overlap = jStat.dot(first, second);
  for (var i = 0; i < second.length; ++i) {
    second[i] -= overlap * first[i];
  }
  normalize(second);

  for (var i = 0; i < n_t; ++i) {
    var angle = 2 * Math.PI * i / n_t;
    var c_first = magnitude * Math.cos(angle);
    var c_second = magnitude * Math.sin(angle);
    for (var j = 0; j < mat[i].length; ++j) {
      mat[i][j] = c_first * first[j] + c_second * second[j];
    }
  }
  return mat;
}

// A base class for finite looping Gaussian oscillators, which store all their
// values in memory in the matrix.
function finiteLoopingOscillator(matrix) {
  var noise = looper(matrix);

  return Object.assign(
      Object.create(independentOscillator(matrix[0].length)),
      {
        advance: function() {
          noise.advance();
        },
        currentNoise: function() {
          return noise.current();
        }
      });
}

// Great Circle oscillators (Hennig 2013).
function greatCircleOscillator(n, n_total) {
  return finiteLoopingOscillator(greatCircleMatrix(n, n_total));
}

function OscillatingMatrix(n_indep, n_timesteps) {
  return jStat.create(n_timesteps, 2 * n_indep, function(i, j) {
    // The more independent points, the longer we go before repeating: note that
    // max(t) = 2.0 * n_indep.
    var t = 2.0 * i * n_indep / n_timesteps;
    var trig_func = (j % 2 == 0) ? Math.sin : Math.cos;
    var order = Math.floor(j / 2) + 1;
    return trig_func(Math.PI * order * t / n_indep) / Math.sqrt(n_indep);
  });
}

// Delocalized oscillators (Hogg 2013).
function delocalizedOscillator(n, n_t, n_indep) {
  return finiteLoopingOscillator(
      jStat(OscillatingMatrix(n_indep, n_indep * n_t))
      .multiply(jStat.create(2 * n_indep, n, standardNormal)));
}

// Thanks to http://stackoverflow.com/a/10284006 for zip() function.
function zip(arrays) {
  return arrays[0].map(function(_, i) {
    return arrays.map(function(array) { return array[i]; })
  });
}

// Compute a new upper-triangular cholesky root, given a covariance function.
function upperCholeskyCovariance(x, kFunc) {
  return jStat.transpose(Cholesky(CovarianceMatrix(x, kFunc)));
}

function DatasetGenerator(x, mu, kFunc, n_t) {
  // The upper-triangular cholesky root of the (space) covariance matrix.
  var U = upperCholeskyCovariance(x, kFunc);

  return {
    // The x-values for this closure.
    x: x,
    // The number of points in the dataset.
    N: x.length,
    // The underlying Gaussian Oscillator which powers this animation.
    gaussianOscillator: {
      advance: function() {
        alert('Abstract base class!  Must override gaussianOscillator.');
      }
    },
    // Advance to the next dataset and return it.
    // Advance to the next dataset and return it.
    NextDataset: function() {
      this.gaussianOscillator.advance();
      return jStat(this.gaussianOscillator.currentNoise()).multiply(U)[0];
    },
    // Update to a new space-domain covariance.
    UpdateCovariance: function(kFunc) {
      U = upperCholeskyCovariance(this.x, kFunc);
    }
  };
};

function CompactSupportCovarianceGenerator(x, mu, kFunc, n_t) {
  return Object.assign(
      Object.create(DatasetGenerator(x, mu, kFunc, n_t)),
        {
          gaussianOscillator: compactSupportCovarianceOscillator(x.length, n_t)
        });
}

function GreatCircleGenerator(x, mu, kFunc, n_t) {
  return Object.assign(
      Object.create(DatasetGenerator(x, mu, kFunc, n_t)),
        {
          gaussianOscillator: greatCircleOscillator(x.length, 2 * n_t)
        });
}

function DelocalizedGenerator(x, mu, kFunc, n_t, n_indep) {
  return Object.assign(
      Object.create(DatasetGenerator(x, mu, kFunc, n_t)),
        {
          gaussianOscillator: delocalizedOscillator(x.length, n_t, n_indep)
        });
}

function InterpolatingGenerator(x, mu, kFunc, n_t) {
  return Object.assign(
      Object.create(DatasetGenerator(x, mu, kFunc, n_t)),
        {
          gaussianOscillator: interpolatingOscillator(x.length, n_t)
        });
};

function genericLinearModel(modelFunctions, options) {
  return Object.assign(
      {
        // Whether or not the given x-value is within the bounds (if any) of
        // this model.
        inBounds: function(x) {
          return (this.bounds && this.bounds[0] && this.bounds[1]
                  ? x >= this.bounds[0] && x <= this.bounds[1]
                  : true);
        },

        // An array of function(x).  The i'th element gives the i'th model
        // Function.
        modelFunctions: modelFunctions,

        // The i'th model function's value at x -- but only if x is within the
        // bounds (if any).
        boundedModelFunction: function(x, i) {
          return this.inBounds(x) ? this.modelFunctions[i](x) : 0;
        },

        // The total prediction for the model at x, given parameter values beta.
        modelPrediction: function(x, beta) {
          return beta.map(
              function(v, i) {
                return v * this.boundedModelFunction(x, i);
              }, this).reduce(function(a, b) { return a + b; });
        },

        // Compute a matrix to fit the data at these particular x-values (or at
        // least, the ones which are in-bounds if this model is bounded).
        train: function(x) {
          // Analytical solution for ordinary least squares.
          var X = jStat.create(modelFunctions.length, x.length, (
                function(i, j) {
                  return this.boundedModelFunction(x[j], i);
                }).bind(this));
          this.M = jStat.multiply(
              jStat.inv(jStat.multiply(X, jStat.transpose(X))), X);
          this.xMin = this.bounds && this.bounds[0] || Math.min.apply(null, x);
          this.xMax = this.bounds && this.bounds[1] || Math.max.apply(null, x);
        }
      },
      options);
}

function linearModel(options) {
  return genericLinearModel(
      [function(_) { return 1; },
       function(x) { return x; }],
      options);
}

// A set of data points which can change over time.
//
// Args:
//   options:  A dictionary of options, especially:
//     y:  An array of y-values, assumed to be the same length as x.
//     animatedNoise:  A Gaussian oscillator which defines how the data will
//       change over time.
function animatedDataGenerator(x, options) {
  return Object.assign(
      {
        // Advance to the next data values.
        advance: function() {
          this.animatedNoise && this.animatedNoise.advance &&
            this.animatedNoise.advance();
        },
        
        // Retrieve the current y-value.  Both this.y and this.currentNoise
        // default to 0 if absent.
        currentY: function() {
          return x.map(
              function(_, i) {
                return (this.y && this.y[i] || 0) + 
                       (this.animatedNoise &&
                        this.animatedNoise.currentNoise()[i] || 0);
              }, this);
        },

        // Useful for constructing a google.visualization.DataTable.
        toJSON: function() {
          return {
            cols: [
              {id: 'x', label: 'x', pattern: '', type: 'number'},
              {id: 'y', label: 'y', pattern: '', type: 'number'}],
            rows: zip([x, this.currentY()])
          };
        }
      },
      options);
}

// Return a chart object.
function AnimatedChart(dataset_generator, div_id, title, chart_type, options) {
  chart_type = (typeof chart_type !== 'undefined') ?
    chart_type : google.visualization.LineChart;
  // The generator which generates new datasets.
  var generator = dataset_generator;
  // The number of milliseconds for each frame.
  var frame_length = 250;
  // Copy the x-values for the data.
  var x = generator.x.slice();

  var data = new google.visualization.DataTable();
  data.addColumn('number', 'x');
  data.addColumn('number', 'y');
  data.addRows(zip([x, generator.NextDataset()]));

  // Set chart options.
  var chart_options = $.extend(
      {
        title: title,
        width: 800,
        vAxis: {
          viewWindow: {
            min: -3.0,
            max: 3.0,
          },
        },
        animation: {
          duration: frame_length,
          easing: 'linear',
        },
        height: 500
      },
      options);

  var return_object = {
    animation_id: null,
  };

  return_object.chart = new chart_type(document.getElementById(div_id))
  return_object.chart.draw(data, chart_options);

  return_object.draw = function() {
    // Kick off the animation.
    return_object.chart.draw(data, chart_options);

    // Compute the new data for the next frame.
    var new_data = generator.NextDataset();
    for (var i = 0; i < new_data.length; ++i) {
      data.setValue(i, 1, new_data[i]);
    }
  };

  return_object.UpdateCovariance = function(k) {
    generator.UpdateCovariance(k);
  }

  // Functions to start and stop the animations.
  var listener_id = null;
  return_object.stop = function() {
    if (listener_id !== null) {
      google.visualization.events.removeListener(listener_id);
      listener_id = null;
    }
  }
  return_object.start = function() {
    listener_id = google.visualization.events.addListener(
        return_object.chart, 'animationfinish', return_object.draw);
    return_object.chart.draw(data, chart_options);
  }

  return return_object;
};
