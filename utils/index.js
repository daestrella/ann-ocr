import { NeuralNetwork } from './ann.js';

document.addEventListener('DOMContentLoaded', function() {
    let rcount = {container: document.getElementById('rows'), value: 7};		// number of image row
    let ccount = {container: document.getElementById('cols'), value: 5};		// number of image column

    let hlcount = {container: document.getElementById('hlayers'), value: 2};		// number of hidden layers
    let lrate = {container: document.getElementById('lrate'), value: 0.001};		// learning rate
    let decay = {container: document.getElementById('decay'), value: 0.1};		// learning rate decay
    let min_error = {container: document.getElementById('min-error'), value: 1e-12};	// minimum error during training
    let max_epoch = {container: document.getElementById('max-epoch'), value: 500};	// maximum number of epochs before terminating training
    let loss = 'MSE';

    let network_button = document.getElementById('net-create');

    let inputs = {container: null, value: null};		// input csv file
    let outputs = {container: null, value: null};		// [target] output csv file
    let train_button = null;
    
    const tresults_container = document.getElementById('training-results');
    const table_container = document.getElementById('input-details');
    const output_container = document.getElementById('output-details');

    let network = null;
    let training_inputs = null;	    // array of input combinations
    let training_outputs = null;    // array of outputs (given the inputs)

    for (let element of [rcount, ccount, hlcount, lrate, decay, min_error, max_epoch]) {
	element.container.addEventListener('input', function() {
	    element.value = parseFloat(element.container.value);
	});
    }
    
    network_button.addEventListener('click', function () {
	table_container.innerHTML = '';
	tresults_container.innerHTML = '';
	output_container.innerHTML = '';
	if (train_button) train_button.remove();

	inputs.value = null;
	outputs.value = null;
	network = new NeuralNetwork([(rcount.value + ccount.value), hlcount.value], lrate.value, decay.value);

	const training_div = document.getElementById('training-details');
	training_div.innerHTML = `
	    <h2>Training dataset</h2>
	    <label for="inputs">Select a input training data:</label>
	    <input type="file" id="inputs" name="inputs" accept=".csv" />
	    <br>
	    <label for="outputs">Select a output training data:</label>
	    <input type="file" id="outputs" name="outputs" accept=".csv" />
	`;

	inputs.container = document.getElementById('inputs');
	outputs.container = document.getElementById('outputs');

	inputs.container.addEventListener('change', async function() {
	    const file = inputs.container.files[0];

	    if (file) {
		try {
		    inputs.value = await parseCSV(file)	//must be a 2D array
		} catch (error) {
		    console.error('Error parsing CSV: ', error);
		}
	    }
	});

	outputs.container.addEventListener('change', async function() {
	    const file = outputs.container.files[0];

	    if (file) {
		try {
		    outputs.value = await parseCSV(file)	//must be a 2D array
		} catch (error) {
		    console.error('Error parsing CSV: ', error);
		}
	    }
	});

	train_button = document.createElement('button');
	train_button.id = 'train-button';
	train_button.type = 'button';
	train_button.innerHTML = 'Train network';

	train_button.addEventListener('click', function() {
	    trainNetwork();
	});

	training_div.insertAdjacentElement('afterend', train_button);
    });

    function parseCSV(file) {
	return new Promise((resolve, reject) => {
	    const reader = new FileReader();
	    let data = [];

	    reader.onload = function(event) {
		const csv_data = event.target.result;
		resolve(csv_data.split('\n')
		    .filter(row => row.trim() !== '')		// removes empty lines
		    .map(row => row.split(',').map(Number)));	//result is a 2D array
	    };

	    reader.onerror = function() {
		reject(reader.error);
	    };
	
	    reader.readAsText(file);
	});
    }

    async function trainNetwork() {
	if (!inputs.value || !outputs.value) {
	    throw new TypeError('Incomplete input or output');	// Ensures there exists both input and output data
	}
	
	if (!isValid(inputs.value, outputs.value)) {
	    throw new TypeError('Input or output not valid.');
	}

	tresults_container.innerHTML = `
	    <h2>Network training details</h2>
	    <p>Currently training the network...</p>
	`;

	const training_details = await network.train(inputs.value, outputs.value, min_error.value, max_epoch.value);

	showErrorGraph(training_details);
	createTable();
    }

    function isValid(input_array, output_array) {
	console.log(input_array, output_array);
	//console.log('input-output shape equality: ', output_array.length === input_array.length);
	//console.log('consistent input rows/columns: ', input_array.every(row => row.length === (rcount.value + ccount.value)));
	//console.log('consistent output columns: ', output_array.every(row => row.length === 1));
	return ((output_array.length === input_array.length) &&					    // validates equality of input and output rows
		 input_array.every(row => row.length === (rcount.value + ccount.value)) &&
		 output_array.every(row => row.length === 1));
    }

    function showErrorGraph(tdetails) {
	tresults_container.innerHTML = `
	    <h2>Network training details</h2>
	    <canvas id='error-chart'></canvas>
	`;

        let error_chart = {container: document.getElementById('error-chart').getContext('2d'), value: null};

        // Create the initial chart
        error_chart.value = new Chart(error_chart.container, {
            type: 'line',
            data: {
                labels: Array.from({ length: tdetails.epochs }, (_, i) => (i + 1).toString()),
                datasets: [{
                    label: `Network error (${loss}) per epoch`,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                    data: tdetails.errors,
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true // Start y-axis from zero
                    }
                }
            }
        });
    }

    function createTable() {
	table_container.innerHTML = `
	    <h2>Network input</h2>
	    <p>Input image to be read.</p>
	`;

	const table = document.createElement('table');
	table.id = 'toggleable-table';

	for (let i = 0; i < rcount.value; i++) {
	    let row = document.createElement('tr');
	    for (let j = 0; j < ccount.value; j++) {
		let cell = document.createElement('td');
		cell.id = 'white-cell';
		cell.addEventListener('click', function() {
		    toggleCell(cell);
		});
		row.appendChild(cell);
	    }
	    table.appendChild(row);
	}
	const pred_button = document.createElement('button');
	pred_button.id = 'predict-button';
	pred_button.type = 'button';
	pred_button.innerHTML = 'Predict';

	table_container.appendChild(table);
	table_container.appendChild(pred_button);

	pred_button.addEventListener('click', function() {
	    let inputs = parseTable()
	    let output = network.predict(inputs);
	    output_container.innerHTML = `
		<h2>Network output</h2>
		<p>${inputs}</p>
		<p>${String.fromCharCode(output)} (${output})</p>
	    `;
	});
    }

    function toggleCell(cell) {
	if (cell.id === 'white-cell') {
	    cell.id = 'black-cell';
	} else {
	    cell.id = 'white-cell';
	}
    }

    function parseTable() {
	const table = document.getElementById('toggleable-table');
	
	let white_per_row = new Array(table.rows.length).fill(0);
	let white_per_col = new Array(table.rows[0].cells.length).fill(0);

	for (let i = 0; i < table.rows.length; i++) {
	    let row = table.rows[i];
	    for (let j = 0; j < table.rows[0].cells.length; j++) {
		let cell = row.cells[j];
		if (cell.id === 'white-cell') {
		    white_per_row[i]++;
		    white_per_col[j]++;
		}
	    }
	}

	return white_per_row.concat(white_per_col);
    }
});

