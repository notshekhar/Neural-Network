function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}


function dsigmoid(y) {
  // return sigmoid(x) * (1s - sigmoid(x));
  return y * (1 - y);
}

// function tanh(x) {
//   var y = Math.tanh(x);
//   return y;
// }
//
// function dtanh(x) {
//   var y = 1 / (pow(Math.cosh(x), 2));
//   return y;
// }


class fnn {
  constructor(arr) {
    this.neurons = [];
    this.weights = [];

    let arrlen = arr.length;
    for(let i=0; i<arrlen; i++){
      this.neurons.push(arr[i]);
    }
    for(let i=0; i<arrlen-1; i++){
      let weight = new Matrix(this.neurons[i+1], this.neurons[i]);
      weight.randomize();
      this.weights.push(weight);
    }

  }
  query(inputarr){
    let inputs = Matrix.fromArray(inputarr);
    let weightlen = this.weights.length;
    for(let i=0; i<weightlen; i++){
      inputs = Matrix.multiply(this.weights[i], inputs);
      inputs.map(sigmoid);
    }
    let output = inputs.toArray();
    return output;

  }

  learn(input, outputarr){
        let inputs = Matrix.fromArray(input);
        let outputs = this.query(inputs);
        let answer = Matrix.fromArray(outputarr);
        let err = answer-outputs;
        let errors = [];
        //need to work to be continue

  }
  download(filename){
    let arr = {
      "weights": this.weights
    }

    let datStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(arr));
    let downloadNode = document.createElement("a");
    downloadNode.setAttribute("href", datStr);
    downloadNode.setAttribute("download", filename + ".json");
    downloadNode.click();
    downloadNode.remove();

  }

}


// [password = 48616813]
// [application no = 2018178292]
