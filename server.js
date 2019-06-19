var NeuralNetwork = require('./src/NeuralNetwork')

const neural_network = new NeuralNetwork()

console.log("Random starting synaptic weights: ")
console.log(neural_network.synaptic_weights)

// The training set. We have 4 examples, each consisting of 3 input values
// and 1 output value.
var training_set_inputs = [
    [0, 0, 1],
    [2, 2, 1],
    [2, 2, 4],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
]
// chuyen thanh ma tran cot
var training_set_outputs = neural_network.transpose([
    [0, 0.5, 0.5, 1, 1, 0]
])
console.log("TCL: training_set_outputs", training_set_outputs)

// Train the neural network using a training set.
//Do it 10,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 10000)

console.log("New synaptic weights after training: ")
console.log(neural_network.synaptic_weights)

// Test the neural network with a new situation.
console.log("Considering new situation [1, 0, 0] -> ?: ")
console.log(neural_network.think([[1, 0, 0],[2, 2, 3]]))