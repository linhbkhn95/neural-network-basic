class NeuralNetwork {
  constructor() {
    this.synaptic_weights = [[-0.16595599], [0.44064899], [-0.99977125]];
  }
  matrixDot(A, B) {
    ////console.log("TCL: NeuralNetwork -> matrixDot -> A", A)
    ////console.log("TCL: NeuralNetwork -> matrixDot -> B", B)

    var result = new Array(A.length)
      .fill(0)
      .map(row => new Array(B[0].length).fill(0));

    return result.map((row, i) => {
      return row.map((val, j) => {
        return A[i].reduce((sum, elm, k) => sum + elm * B[k][j], 0);
      });
    });
  }
  transpose(matrix) {
    return matrix.reduce(
      (prev, next) => next.map((item, i) => (prev[i] || []).concat(next[i])),
      []
    );
  }
  //ham mat mat
  __sigmoid(x) {
    for (var i = 0; i < x.length; i++) {
      for (var j = 0; j < x[i].length; j++) {
        x[i][j] = 1 / (1 + Math.exp(-x[i][j]));
      }
    }
    return x;
  }
  add(a, b) {
    var rs = [];
    var rows = [];
    var item = 0;
    for (var i = 0; i < a.length; i++) {
      rows = [];

      for (var j = 0; j < a[i].length; j++) {
        item = a[i][j] + b[i][j];
        rows.push(item);
      }
      rs.push(rows);
    }
    return rs;
  }
  sub(a, b) {
    var rs = [];
    var rows = [];
    var item = 0;
    for (var i = 0; i < a.length; i++) {
      rows = [];

      for (var j = 0; j < a[i].length; j++) {
        item = a[i][j] - b[i][j];
        rows.push(item);
      }
      rs.push(rows);
    }
    return rs;
  }
  __sigmoid_derivative(x) {
    for (var i = 0; i < x.length; i++) {
      for (var j = 0; j < x[i].length; j++) {
        x[i][j] = x[i][j] * (1 - x[i][j]);
      }
    }
    //console.log("TCL: NeuralNetwork -> __sigmoid_derivative -> x", x);

    return x;
    // return x * (1 - x)
  }
  train(training_set_inputs, training_set_outputs,number_of_training_iterations){
    var error, adjustment, output, err;
    for (var i = 0; i < number_of_training_iterations; i++) {
      // Pass the training set through our neural network (a single neuron).
      output = this.think(training_set_inputs);
      //console.log("TCL: NeuralNetwork -> train -> output", output);

      // Calculate the error (The difference between the desired output
      // and the predicted output).
      // error =  training_set_outputs - output
      error = this.sub(training_set_outputs, output);
      //console.log("TCL: NeuralNetwork -> train -> error", error);

      // Multiply the error by the input and again by the gradient of the Sigmoid curve.
      // This means less confident weights are adjusted more.
      // This means inputs, which are zero, do not cause changes to the weights.
      err = this.matrixDot(error, this.__sigmoid_derivative(output));
      //console.log("TCL: NeuralNetwork -> err", err);
      adjustment = this.matrixDot(this.transpose(training_set_inputs), err);

      //  //console.log("TCL: NeuralNetwork -> adjustment", adjustment);

      // Adjust the weights.
      this.synaptic_weights = this.add(this.synaptic_weights, adjustment);
    }
  }
  think(inputs) {
    // Pass inputs through our neural network (our single neuron).
    let dotrs = this.matrixDot(inputs, this.synaptic_weights);
    ////console.log("TCL: NeuralNetwork -> think -> dotrs", dotrs);
    var rs = this.__sigmoid(dotrs);
    ////console.log("TCL: NeuralNetwork -> think -> rs", rs);
    return rs;
  }
}

module.exports = NeuralNetwork;
