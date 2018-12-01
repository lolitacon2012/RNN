//utils
import math from "mathjs";
import corpus from "nlp-corpus";
import argmax from 'compute-argmax';

const mulGate = new MultiplyGate()
const addGate = new AddGate()
const activation = new Tanh()

function argMax(array) {
  return argmax(array)[0];
}

function sample(array){
	const ran = Math.random();
	let pSum = 0;
	let result = 0;
	for(let i = 0 ; i< array.length;i++){
		pSum+=array[i];
		if(ran<=pSum){
			return i;
		}
	}
}

function MultiplyGate(){
	this.forward = (W, x) => {
		return math.multiply(W, x);
	};
	this.backward = (W, x, dz) => {
		const dW = math.multiply(dz, math.transpose(x));
		const dx = math.multiply(math.transpose(W), dz);
		return [dW, dx]
	}
}

function AddGate(){
	this.forward = (x, y) => {
		return math.add(x, y);
	};
	this.backward = (x1, x2, dz) => {
		const dx1 = math.dotMultiply(math.ones(x1.size()),dz);
		const dx2 = math.dotMultiply(math.ones(x2.size()),dz);
		return [dx1, dx2]
	}
}

function SigmoidGate(){
	this.forward = (x) => {
		return 1/(1 + Math.pow(Math.E, 0-x))
	};
	this.backward = (x, diff) => {
		const output = this.forward(x);
		return (1 - output) * output * diff;
	}
}

function Tanh(){
    this.forward = (x) => {
        return math.tanh(x)
    }
    this.backward = (x, diff) => {
		const output = this.forward(x);
		return math.dotMultiply(math.multiply(math.add(math.square(output), -1), -1),diff);
    }
}

function Softmax(){
	this.predict = (x)=> {
		const exp_scores = math.exp(x)
		return math.divide(exp_scores, math.sum(exp_scores));
	}
	this.loss = (x, index) => {
		const probs = this.predict(x);
		return -math.log(probs.get([index,0]))
	}
	this.diff = (x, index) => {
		let probs = this.predict(x);
		const original_value = probs.get([index,0])
		probs.set([index,0], original_value - 1);
		return probs;
	}
}

function RNNLayer(){
	this.mulu = undefined;
	this.mulw = undefined;
	this.add = undefined;
	this.s = undefined;
	this.mulv = undefined;
	this.forward = (x, prev_s, U, W, V) => {
		this.mulu = mulGate.forward(U, x)
		this.mulw = mulGate.forward(W, prev_s)
        this.add = addGate.forward(this.mulw, this.mulu)
        this.s = activation.forward(this.add)
		this.mulv = mulGate.forward(V, this.s)
	}
	this.backward = (x, prev_s, U, W, V, diff_s, dmulv)=>{
		this.forward(x, prev_s, U, W, V);
		const [dV, dsv] = mulGate.backward(V, this.s, dmulv)
		const ds = math.add(dsv, diff_s);
		const dadd = activation.backward(this.add, ds);
		const [dmulw, dmulu] = addGate.backward(this.mulw, this.mulu, dadd);
		const [dW, dprev_s] = mulGate.backward(W, prev_s, dmulw)
		const [dU, dx] = mulGate.backward(U, x, dmulu)
        return [dprev_s, dU, dW, dV]
	}
}

function Model(){
	this.word_dim = undefined; //size of vocabulary
    this.hidden_dim = undefined; //size of hidden layer
    this.bptt_truncate = undefined;
    this.U = undefined;
    this.W = undefined;
    this.V = undefined;
    this.dU = undefined;
    this.dW = undefined;
    this.dV = undefined;
	this.initialize = (word_dim, hidden_dim=100, bptt_truncate=4, preset) => {
		this.word_dim = word_dim;
        this.hidden_dim = hidden_dim;
        this.bptt_truncate = bptt_truncate;
        this.U = [];
        this.W = [];
        this.V = [];
        for(let i=0;i<hidden_dim;i++){
        	this.U.push([]);
        	for(let j=0;j<word_dim;j++){
        		const r = (Math.random() * Math.sqrt(1/word_dim) * 2) - Math.sqrt(1/word_dim);
        		this.U[i].push(r)
        	}
        }
        for(let i=0;i<hidden_dim;i++){
        	this.W.push([]);
        	for(let j=0;j<hidden_dim;j++){
        		const r = (Math.random() * Math.sqrt(1/hidden_dim) * 2) - Math.sqrt(1/hidden_dim);
        		this.W[i].push(r)
        	}
        }
        for(let i=0;i<word_dim;i++){
        	this.V.push([]);
        	for(let j=0;j<hidden_dim;j++){
        		const r = (Math.random() * Math.sqrt(1/hidden_dim) * 2) - Math.sqrt(1/hidden_dim);
        		this.V[i].push(r)
        	}
		}
		this.U = math.matrix(this.U);
		this.W = math.matrix(this.W);
		this.V = math.matrix(this.V);
	}
	this.forward_propagation = (x) => {
		//Here x is a sentence.
		let total_t = x.length;
		let layers = [];
		let prev_s = math.zeros(this.hidden_dim,1);
		for(let t=0;t<total_t;t++){
			const layer = new RNNLayer();
			let input = math.zeros(this.word_dim,1);
			input.set([x[t],0], 1);
			layer.forward(input, prev_s, this.U, this.W, this.V);
			prev_s = layer.s;
			layers.push(layer)
		}
		return layers
	}
	this.predict = (x, strictArgMax) => {
		const output = new Softmax();
		const layers = this.forward_propagation(x);
		return layers.map((layer)=>{
			if(strictArgMax){
				return argMax(math.squeeze(output.predict(layer.mulv)).toArray());
			}else{
				return sample(math.squeeze(output.predict(layer.mulv)).toArray());
			}
		})
	}
	this.calculate_loss = (x, y) => {
		const output = new Softmax();
		const layers = this.forward_propagation(x);
		let loss = 0;
		layers.forEach((layer,index) => {
			loss += output.loss(layer.mulv, y[index])
		})
		return loss / y.length;
	}
	this.calculate_total_loss = (X_ALL, Y_ALL) => {
		let loss = 0;
		for(let i=0;i<Y_ALL.length;i++){
			loss += this.calculate_loss(X_ALL[i], Y_ALL[i])
		}
		return loss / Y_ALL.length;
	}
	this.bptt = (x, y) => {
		const output = new Softmax();
		const layers = this.forward_propagation(x);
    	const total_t = layers.length;
    	let prev_s_t = math.zeros(this.hidden_dim,1)
    	let diff_s = math.zeros(this.hidden_dim,1)
    	let dmulv;
    	let input;
    	let dprev_s, dU_t, dW_t, dV_t, dU_i, dW_i, dV_i;
    	let dV = math.zeros(math.size(this.V))
    	let dU = math.zeros(math.size(this.U))
    	let dW = math.zeros(math.size(this.W))

    	for(let t = 0 ; t<total_t;t++){
    		dmulv = math.matrix(output.diff(layers[t].mulv, y[t]))
        	input = math.zeros(this.word_dim,1)
        	input.set([x[t],0], 1);
        	[dprev_s, dU_t, dW_t, dV_t] = layers[t].backward(input, prev_s_t, this.U, this.W, this.V, diff_s, dmulv)
        	prev_s_t = layers[t].s
        	dmulv = math.zeros(this.word_dim,1)
    		for(let i=t-1;i>Math.max(-1, t-this.bptt_truncate-1);i--){
    			input = math.zeros(this.word_dim,1)
    			input.set([x[i],0], 1);
				const prev_s_i = (i == 0) ? math.zeros(this.hidden_dim,1) : layers[i-1].s
				const layer_result = layers[i].backward(input, prev_s_i, this.U, this.W, this.V, dprev_s, dmulv);
				[dprev_s, dU_i, dW_i, dV_i] = layer_result
				dU_t = math.add(dU_i, dU_t)
        	    dW_t = math.add(dW_i, dW_t)
    		}
    		dV = math.add(dV_t, dV)
        	dU = math.add(dU_t, dU)
        	dW = math.add(dW_t, dW)
        	//console.log("training prediction: " + x[t] + " -> " + y[t])
    	}
    	return [dU, dW, dV];
	}
	this.sgd_step = (x, y, learning_rate) => {
		let [dU, dW, dV] = this.bptt(x, y)
		this.U = math.add(this.U, math.multiply( -learning_rate, dU))
    	this.V = math.add(this.V, math.multiply( -learning_rate, dV))
    	this.W = math.add(this.W, math.multiply( -learning_rate, dW))
	}
	this.train = (X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5) => {
		let num_examples_seen = 0
    	let losses = []
    	for(let epoch = 0; epoch < nepoch; epoch ++){
    		if (epoch % evaluate_loss_after == 0){
    			const loss = this.calculate_total_loss(X, Y)
            	losses.push([num_examples_seen, loss])
            	//const time = new Date();
            	console.log(`Loss after num_examples_seen=${num_examples_seen} epoch=${epoch}: ${loss}`);
            	if (losses.length > 1 && losses[losses.length-1][1] > losses[losses.length-2][1]){
                	learning_rate = learning_rate * 0.5
                	console.log("Setting learning rate to " + learning_rate)
            	}
    		}
    		Y.forEach((y, index)=>{
            	this.sgd_step(X[index], y, learning_rate)
            	num_examples_seen += 1
    		});
    	}
    	return losses;
	}
}

// prepare dictionary
let indexToWord = new Map();
let wordToIndex = new Map();
indexToWord.set(0, "##START");
indexToWord.set(1, "##END");
wordToIndex.set("##START", 0);
wordToIndex.set("##END", 1);
let full_text = [0];
const generated_training_set = [`jiang ze min`]
//corpus.generate(2000);

generated_training_set.join('\n').split(" ").map((s)=>{
	s.replace(/[.,<>/?!@"{}\n;:=+\-_)(*&^%$@)]/g, (a)=>{
		switch(a){
			case ".":
			case "?":
			case "!":
			case "\n":
			return " " + a + " ##END ##START "
		}
		return " " + a + " "
		// s.replace(/[\n]/g, (a)=>{
		// 	return " ##END ##START "
		}).split(" ").forEach((ss)=>{
		if(ss.length>0){
			const index = wordToIndex.get(ss);
			if(index!=undefined){
				full_text.push(index);
			}else{
				const new_index = indexToWord.size;
				indexToWord.set(new_index, ss);
				wordToIndex.set(ss, new_index);
				full_text.push(new_index);
			}
		}
	})
});

let X_train = [];
let Y_train = [];
let pivot = 0;
let sentenceLength = 8;
while(true){
	if(pivot+sentenceLength>=full_text.length){
		break;
	}
	const X_train_single = [];
	const Y_train_single = [];
	X_train_single.push(full_text[pivot])
	for(let i=1;i<sentenceLength;i++){
		X_train_single.push(full_text[pivot+i])
		Y_train_single.push(full_text[pivot+i])
	}
	Y_train_single.push(full_text[pivot+sentenceLength])
	X_train.push(X_train_single)
	Y_train.push(Y_train_single)
	pivot+=Math.floor(sentenceLength/2);
}

//start training!
const hidden_dim = 20;
const word_dim = indexToWord.size;
const rnn = new Model();
console.log(`
	word size = ${word_dim}`)
rnn.initialize(word_dim, hidden_dim);
rnn.train(X_train,Y_train,1,120,1);

const generate_length_max = 500;
let output_full = [0];
for(let m=0;m<generate_length_max;m++){
	const step = rnn.predict([output_full[m]]);
	output_full.push(step[0]);
}
output_full = (output_full.map((e)=>{
	if(e<2){
		return "";
	}
	return indexToWord.get(e);
}).slice(1))
let no_space = true;
console.log ( "RESULT = \n" + output_full.map((e)=>{
	let r = "";
	switch(e){
		case ".":
		case ",":
		case ":":
		case ";":
		case "?":
		case "!":
		case ")":
		case "]":
			r = e + " ";
			no_space = true;
			break;
		case "\n":
		case "":
			r = "";
		default:
			r = no_space ? e : (" " + e);
			if(no_space){
				no_space = false;
			}
	}
	return r;
}).join(""));


