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

// Thanks to http://stackoverflow.com/a/10284006 for zip() function.
function zip(arrays) {
  return arrays[0].map(function(_, i) {
    return arrays.map(function(array) { return array[i]; })
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

function OscillatingMatrix(n_indep, n_timesteps) {
  var factor = 1.0 / Math.sqrt(n_indep);
  return jStat.create(n_timesteps, 2 * n_indep, function(i, j) {
    // The more independent points, the longer we go before repeating: note that
    // max(t) = 2.0 * n_indep.
    var t = 2.0 * i * n_indep / n_timesteps;
    var trig_func = (j % 2 == 0) ? Math.sin : Math.cos;
    var order = Math.floor(j / 2) + 1;
    return trig_func(Math.PI * order * t / n_indep) / Math.sqrt(n_indep);
  });
}

function DatasetGenerator(x, mu, kFunc, N_t) {
  var return_object = {};

  // First section: declare member variables for the closure.
  //
  // The x-values for this closure.
  return_object.x = x;
  // The number of timesteps we need to keep in memory.
  var N_t = N_t;
  // The number of points in the dataset.
  var N = x.length;
  // The time-domain covariance matrix (with compact support).
  var K_t = CompactSupportCovarianceMatrix(N_t);
  // Each row of L_t is a vector to multiply different timesteps.
  var L_t = LoopingCholesky(K_t);
  var random_matrix = jStat.create(N_t, N, function(i, j) {
    return jStat.normal.sample(0, 1);
  });
  // i indicates which vector from L_t to use, and also which row of the random
  // matrix to update.
  var i = N_t - 1;
  // The covariance matrix in space.
  //
  // We add a small amount of noise on the diagonal to help computational
  // stability.
  var K = CovarianceMatrix(x, kFunc);
  var U = jStat.transpose(Cholesky(K));

  return_object.NextDataset = function() {
    // Compute the next data.
    var independent_data = jStat(L_t[i]).multiply(random_matrix)[0];
    // Generate new random numbers.
    for (var j = 0; j < N; ++j) {
      random_matrix[i][j] = jStat.normal.sample(0, 1);
    }
    // Update the counter.
    i = ((i > 0) ? i : N_t) - 1
    // Return the next dataset.
    var new_data = jStat(independent_data).multiply(U)[0];
    return new_data;
  }

  return_object.UpdateCovariance = function(kFunc) {
    var K = CovarianceMatrix(x, kFunc);
    U = jStat.transpose(Cholesky(K));
  }

  return return_object;
};

function HennigGenerator(x, mu, kFunc, N_t) {
  return OscillatingGeneratorBase(x, mu, kFunc, N_t, 1, true);
}

function OscillatingGenerator(x, mu, kFunc, N_t, N_indep) {
  return OscillatingGeneratorBase(x, mu, kFunc, N_t, N_indep, false);
}

function OscillatingGeneratorBase(x, mu, kFunc, N_t, N_indep, lock_magnitude) {
  var return_object = {};

  // First section: declare member variables for the closure.
  //
  // The x-values for this closure.
  return_object.x = x;
  // The number of timesteps from one "independent" frame to the next.
  var N_t = N_t;
  // The number of points in the dataset.
  var N = x.length;
  // The "time" between frames.
  var dt = 1.0 / N_t;
  // The matrix to convert random noise into oscillating timetraces.
  var n_timesteps = N_indep * N_t;
  var M = OscillatingMatrix(N_indep, n_timesteps);
  // Random data we'll use to seed the animation.
  var random_matrix = jStat(M).multiply(
      jStat.create(2 * N_indep, N, function(i, j) {
        return jStat.normal.sample(0, 1);
      }));
  // Passing lock_magnitude == true and N_indep == 1 means we should make this
  // into Hennig's method, where orbits are constrained to be circular.  We do
  // this by giving the second row of random draws the same magnitude as the
  // first.
  if (lock_magnitude && N_indep == 1) {
    for (var a = 0; a < N; ++a) {
      random_matrix[1][a] = random_matrix[0][a];
    }
  }

  // The covariance matrix in space.
  K = CovarianceMatrix(x, kFunc);
  var U = jStat.transpose(Cholesky(K));

  var i = n_timesteps - 1;

  return_object.NextDataset = function() {
    // Compute the next data.
    var independent_data = random_matrix[i];
    // Update the counter.
    i = ((i > 0) ? i : n_timesteps) - 1
    // Return the next dataset.
    var new_data = jStat(independent_data).multiply(U)[0];
    return new_data;
  }

  return_object.UpdateCovariance = function(kFunc) {
    var K = CovarianceMatrix(x, kFunc);
    U = jStat.transpose(Cholesky(K));
  }

  return return_object;
};

function InterpolatingGenerator(x, mu, kFunc, N_t) {
  var return_object = {};

  // First section: declare member variables for the closure.
  //
  // The x-values for this closure.
  return_object.x = x;
  // The number of timesteps from one "independent" frame to the next.
  var N_t = N_t;
  // The number of points in the dataset.
  var N = x.length;
  // Random data we'll use to seed the animation.
  var random_matrix = jStat.create(2, N, function(i, j) {
    return jStat.normal.sample(0, 1);
  });

  // The covariance matrix in space.
  K = CovarianceMatrix(x, kFunc);
  var U = jStat.transpose(Cholesky(K));

  // The number of timesteps after a keyframe.
  var i = 0;
  // Which row is the "primary" row.
  var row = 0;
  // A coefficient matrix to mix the keyframes.
  var coefficients = jStat.create(1, 2, function(i, j) { return 0; });

  return_object.NextDataset = function() {
    var frac = i / N_t;
    // Compute the next data.
    coefficients[row    ] = Math.cos(Math.PI * frac / 2.0);
    coefficients[1 - row] = Math.sin(Math.PI * frac / 2.0);
    var independent_data = jStat(coefficients).multiply(random_matrix)[0];
    var new_data = jStat(independent_data).multiply(U)[0];
    // Update the counter.
    i++;
    if (i == N_t) {
      for (var j = 0; j < N; ++j) {
        random_matrix[row][j] = jStat.normal.sample(0, 1);
      }
      i = 0;
      row = 1 - row;
    }
    // Return the next dataset.
    return new_data;
  }

  return_object.UpdateCovariance = function(kFunc) {
    var K = CovarianceMatrix(x, kFunc);
    U = jStat.transpose(Cholesky(K));
  }

  return return_object;
};

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
