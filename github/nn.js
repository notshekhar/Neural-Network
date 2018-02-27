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
    this.lr = 0.01;
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

    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);


    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);
    //storing all biases
    let bias = [];
    for(let i=0; i<this.bias_h.length; i++){
      bias.push(this.bias_h[i]);
    }
    bias.push(this.bias_o);
    // console.log(bias);
    // stoing all outputs
    let output = [];
    output.push(inputs);
    for(let i=0; i<hidden.length; i++){
      output.push(hidden[i]);
    }
    output.push(outputs);
    // all output transposes
    let output_T = [];
    for (let i = 0; i < output.length; i++) {
      let prott = Matrix.transpose(output[i]);
      output_T.push(prott);
    }

    // console.log(output);
    // storing all weights
    let weights = [];
    weights.push(this.weights_ih);
    for (let i = 0; i < this.weights_hh.length; i++) {
      weights.push(this.weights_hh[i]);
    }
    weights.push(this.weights_ho);
    //weights transpose
    let weights_T = [];
    for(let i = 0; i<weights.length; i++){
      let wt = Matrix.transpose(weights[i]);
      weights_T.push(wt);
    }
    // console.log(weights);

    let errors = [];
    errors.push(output_errors);
    var err = output_errors;

    for(let i=weights_T.length-1; i>=0; i--){
      err = Matrix.multiply(weights_T[i], err);
      errors.push(err);
    }
    errors = errors.reverse();
    // calculating gradients
    let gradients = [];
    for(let i = output.length-1; i>0; i--){
        let grad = Matrix.map(output[i], dsigmoid);
        grad.multiply(errors[i]);
        grad.multiply(this.lr);
        gradients.push(grad);

    }
    gradients = gradients.reverse();
    let dweights = [];
    for(let i=0; i<gradients.length; i++){
      let dw = Matrix.multiply(gradients[i], output_T[i]);
      dweights.push(dw);

    }

    //updating weights

     for(let i=0; i<weights.length; i++){
       weights[i] = weights[i].add(dweights[i]);
     }
    //updating bias
    for(let i=0; i<bias.length; i++){
      bias[i] = bias[i].add(gradients[i]);
    }
    this.weights_ih = weights[0];
    let list = [];
    for (let i = 1; i < weights.length-1; i++) {
      list.push(weights[i]);
    }
    this.weights_hh = list;
    this.weights_ho = weights[weights.length-1];
    let biaslist = [];
    for (let i = 0; i < bias.length-1; i++) {
      biaslist.push(bias[i]);
    }
    this.bias_h = biaslist;
    this.bias_o = bias[bias.length-1];
    output_errors.print();
    // for(let i = 0; i<gradients.length; i++){
    //   let dw = Matrix.multiply(gradients[i], output_T)
    // }
    // outputs.print();
    // targets.print();
    // error.print();
  }

  download(filename){
    let arr = {
      "weights_ih": this.weights_ih,
      "weights_hh": this.weights_ih,
      "weights_ho": this.weights_ih,
      "bias_h": this.weights_ih,
      "bias_o": this.weights_ih
    }

    let datStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(arr));
    let downloadNode = document.createElement("a");
    downloadNode.setAttribute("href", datStr);
    downloadNode.setAttribute("download", filename + ".json");
    downloadNode.click();
    downloadNode.remove();


  }

}
