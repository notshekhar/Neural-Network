var hiddenput;
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}


function dsigmoid(y) {
  // return sigmoid(x) * (1s - sigmoid(x));
  return y * (1 - y);
}




class fnn {
  constructor(arr) {
    // arr will be like [2,3,4,33,1]

    //nodes
    this.input_nodes = arr[0];
    this.hidden_nodes = [];
    //putting all hidden nodes
    for(let i=1; i<arr.length-1; i++){
      this.hidden_nodes.push(arr[i]);
    }

    this.output_nodes = arr[arr.length-1];
    //weights
    this.weights_ih = new Matrix(this.hidden_nodes[0], this.input_nodes);
    this.weights_ih.randomize();

    this.weights_hh = [];
    // putting all hidden weights
    for(let i=1; i<this.hidden_nodes.length; i++){
      var weighthhput = new Matrix(arr[i+1], arr[i]);
      weighthhput.randomize();
      this.weights_hh.push(weighthhput);
    }

    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes[this.hidden_nodes.length-1]);
    this.weights_ho.randomize();


    this.bias_h = [];
    //putting all hidden bias values
    for (let i = 0; i < this.hidden_nodes.length; i++) {
        let biasput = new Matrix(this.hidden_nodes[i], 1);
        biasput.randomize();
        this.bias_h.push(biasput);
    }

    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_o.randomize();
    this.lr = 0.1;
  }


  query(input_array) {


    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = [];
    hiddenput = Matrix.multiply(this.weights_ih, inputs);

    hiddenput.add(this.bias_h[0]);
    hiddenput.map(sigmoid);
    hidden.push(hiddenput);
    // activation function!
    for (let i = 0; i < this.weights_hh.length; i++) {

      hiddenput = Matrix.multiply(this.weights_hh[i], hiddenput);
      hiddenput.add(this.bias_h[i+1]);
      hiddenput.map(sigmoid);
      hidden.push(hiddenput);

    }
    // Generating the output's output!
    let output = Matrix.multiply(this.weights_ho, hidden[hidden.length-1]);
    output.add(this.bias_o);

    output.map(sigmoid);


    // Sending back to the caller!
    return output.toArray();

  }


  learn(input_array, target_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = [];
    hiddenput = Matrix.multiply(this.weights_ih, inputs);

    hiddenput.add(this.bias_h[0]);
    hiddenput.map(sigmoid);
    hidden.push(hiddenput);
    // activation function!
    for (let i = 0; i < this.weights_hh.length; i++) {

      hiddenput = Matrix.multiply(this.weights_hh[i], hiddenput);
      hiddenput.add(this.bias_h[i+1]);
      hiddenput.map(sigmoid);
      hidden.push(hiddenput);

    }
    // Generating the output's output!
    let outputs = Matrix.multiply(this.weights_ho, hidden[hidden.length-1]);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);
    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);


    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);


    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.lr);

    // Calculate deltas
    let hidden_T =[];
    for(let i=0; i<hidden.length; i++){
    let hiddentput = Matrix.transpose(hidden[i]);
    hidden_T.push(hiddentput);
    }

    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T[hidden_T.length-1]);
    // Adjust the weights by deltas
    this.weights_ho.add(weight_ho_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_o.add(gradients);

    //update hidden weights and hidden bias



    // Calcuate input->hidden deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient[0], inputs_T);


    this.weights_ih.add(weight_ih_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
  for (let i = 0; i < hidden_gradient; i++) {
      this.bias_h[i].add(hidden_gradient[i]);
  }


    // outputs.print();
    // targets.print();
    // error.print();
  }


}
