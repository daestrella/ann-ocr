const activation_functions = {
    'ReLU': {
        'function': x => math.map(x, val => math.max(0, val)),
        'derivative': x => math.map(x, val => val >= 0 ? 1 : 0)
    },
    'Leaky ReLU': {
        'function': x => math.map(x, val => math.max(0.01 * val, val)),
        'derivative': x => math.map(x, val => val > 0 ? 1 : 0.01)
    },
    'Identity': {
        'function': x => x,
        'derivative': x => math.ones(math.size(x))
    },
    'Binary': {
        'function': x => math.map(x, val => val > 0 ? 1 : 0),
        'derivative': x => math.zeros(math.size(x))
    },
    'Softplus': {
        'function': x => math.map(x, val => math.log(math.add(1, math.exp(val)))),
        'derivative': x => math.map(x, val => math.divide(1, math.add(1, math.exp(math.unaryMinus(val)))))
    },
    'Tansig': {
        'function': x => math.map(x, val => math.tanh(val)),
        'derivative': x => math.map(x, val => math.subtract(1, math.square(math.tanh(val))))
    }
};

const loss_functions = {
    'MSE': {
        'function': values => math.mean(values.map(([target, actual]) => (target - actual) ** 2)),
        'derivative': values => math.multiply(2, math.mean(values.map(([target, actual]) => actual - target)))
    }
};

const init_methods = {
    'Xavier Uniform': (i, o) => math.random(-Math.sqrt(6)/(i + o), Math.sqrt(6)/(i + o)),
    //'Xavier Normal': (i, o) => math.random('normal', 0, Math.sqrt(2/(i + o))),
    'He Uniform': (i, o) => math.random(-Math.sqrt(6)/i, Math.sqrt(6)/i),
    //'He Normal': (i, o) => math.random([i, o], 'normal', 0, Math.sqrt(2/i))
};

export class NeuralNetwork {
    constructor(structure, lrate, decay, init = 'He Uniform', loss = 'MSE', activation = 'Leaky ReLU') {
	// where struct = [input_neurons, hlayer_count]
	// Note: Architecture assumes a square-shaped network,
	// i.e., count(hlayer_neurons) == count(input_neurons),
	// the output layer is composed of a single neuron,
	// and that the number of hidden layers is 2.
	// for simplicity.
	
	this.init = init_methods[init];
	this.activation = activation_functions[activation];
	this.loss = loss_functions[loss];
	this.lrate = lrate
	this.decay = decay

	this.cache = {'x': [], 'x-hat': []};	// caching for backpropagation
	this.layers = {'weights': [], 'biases': [], 'count': structure[1]}; // count == hlayer_count
	this.#initialize_weights(structure);
    }

    #initialize_weights(structure) {

	for (let i = 0; i < structure[1]; i++) {
	    this.layers.weights.push(
		math.matrix(Array(structure[0]).fill(0).map(() =>
			    Array(structure[0]).fill(0).map(() => this.init(structure[0], 1))))
	    );

	    this.layers.biases.push(
		math.matrix([Array(structure[0]).fill(0).map(() => [this.init(structure[0], 1)])])
	    );
	}

	// output layer weights
	this.layers.weights.push(
	    math.matrix(Array(structure[0]).fill(0).map(() => this.init(structure[0], 1)))
	);

	this.layers.biases.push(
	    math.matrix([this.init(structure[0], 1)])
	);
    }

    predict(inputs, cache = false) {
	// forward passes the inputs and returns a predicted value
	let output = math.matrix(inputs.map(element => [element]));

	for (let i = 0; i <= this.layers.count; i++) {
	    if (cache) this.cache['x'].push(output);   //input for ith layer, output of (i-1)th layer

	    let x_hat = math.add(
		math.multiply(this.layers.weights[i], output),
		this.layers.biases[i])
	    output = this.activation.function(x_hat)._data[0];

	    if (cache) this.cache['x-hat'].push(x_hat._data[0]); //output of ith layer before activation function
	}

	return output;
    }

    train(inputs, outputs, min_error, max_epoch) {
	//trains the network, given the ground truths, and then returns an array of average error per epoch
	let errors = [];

	for (let epoch = 1; epoch <= max_epoch; epoch++) {
	    let error = 0;

	    for (let i = 0; i < inputs.length; i++) {
		let prediction = this.predict(inputs[i], true);
		console.log('Prediction: ', prediction, '; Target: ', outputs[i][0]);
		error += this.loss.function([[outputs[i][0], prediction]]) / inputs.length;    // mean, distributive
		this.backpropagate(outputs[i], prediction, epoch);
	    }

	    errors.push(error);

	    if (isNaN(error)) {
		console.log(`???`);
		return {errors: errors, epochs: epoch};
	    } else if (error <= min_error) {
		console.log(`Finished training after ${epoch} epochs, with ${error/100}% error.`);
		return {errors: errors, epochs: epoch};
	    }
	}

	console.log(`Network did not attain ${min_error/100}% error within ${max_epoch} epochs. [${errors[errors.length-1]/100}% error]`);
	return {errors: errors, epochs: max_epoch};
    }

    backpropagate(target, prediction, epoch) {
	let delta = math.multiply(
	    this.loss.derivative([[target, prediction]]),
	    this.activation.derivative(math.matrix([this.cache['x-hat'][this.layers.count]]))
	)._data[0];

	let lrate = this.lrate * math.exp(-this.decay * epoch);


	for (let i = this.layers.count; i > (this.layers.count - 2); i--) {
	    let bias_gradient = delta;
	    let weights_gradient = math.dotMultiply(bias_gradient, math.transpose(this.cache['x'][i]));

	    this.layers.weights[i] = math.subtract(this.layers.weights[i], math.dotMultiply(lrate, weights_gradient));
	    this.layers.biases[i] = math.subtract(this.layers.biases[i], math.dotMultiply(bias_gradient, lrate));

	    delta = math.dotMultiply(math.transpose(this.layers.weights[i]), delta);
	    delta = math.dotMultiply(delta, this.activation.derivative(this.cache['x-hat'][i-1])); //element-wise multiplication
	};

	for (let i = this.layers.count; i >= 0; i--) {
	    for (let row_delta = 0; row_delta < math.size(delta)[0]; i--) {
		let bias_gradient = math.transpose(row_delta);
		let weights_gradient = math.dotMultiply(bias_gradient, math.transpose(this.cache['x'][i]));

		this.layers.weights[i] = math.subtract(this.layers.weights[i], math.dotMultiply(lrate, weights_gradient));
		this.layers.biases[i] = math.subtract(this.layers.biases[i], math.dotMultiply(lrate, bias_gradient));
	    }
	    
	    if (i !== 0) {
		delta = math.dotMultiply(delta, this.activation.derivative(this.cache['x-hat'][i-1])); //element-wise multiplication
	    }
	}

	this.cache = {'x': [], 'x-hat': []}; //clears cache
    }
}
