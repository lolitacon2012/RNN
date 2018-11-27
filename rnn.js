//utils
const mulGate = new MultiplyGate()
const addGate = new AddGate()
const activation = new Tanh()

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function MultiplyGate(){
	this.forward = (W, X) => {
		return math.multiply(W, X);
	};
	this.backward = (W, X, dz) => {
		const dW = math.multiply(X, dz);
		const dX = math.multiply(W, dz);
		return [dW, dX]
	}
}

function AddGate(){
	this.forward = (X, Y) => {
		return math.add(X, Y);
	};
	this.backward = (X1, X2, dz) => {
		const dX1 = math.multiply(math.ones(X1.length),dz);
		const dX2 = math.multiply(math.ones(X2.length),dz);
		return [dX1, dX2]
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
        return Math.tanh(x)
    }
    this.backward = (x, diff) => {
        const output = this.forward(x);
        return (1 - output * output) * diff;
    }
}

function Softmax(){
	this.predict = (X)=> {
		const exp_scores = math.exp(X)
		return math.divide(exp_scores, math.sum(exp_scores));
	}
	this.loss = (X, index) => {
		const probs = this.predict(X);
		return -math.log(probs[index])
	}
	this.diff = (X, index) => {
		let probs = this.predict(X);
		probs[y] -= 1;
		return probs;
	}
}

function RNNLayer(){
	this.mulu = undefined;
	this.mulw = undefined;
	this.add = undefined;
	this.s = undefined;
	this.mulv = undefined;
	this.forward = (X, prev_s, U, W, V) => {
		this.mulu = mulGate.forward(U, X)
        this.mulw = mulGate.forward(W, prev_s)
        this.add = addGate.forward(this.mulw, this.mulu)
        this.s = activation.forward(this.add)
        this.mulv = mulGate.forward(V, this.s)
	}
	this.backward = (X, prev_s, U, W, V, diff_s, dmulv)=>{
		this.forward(X, prev_s, U, W, V);
		const [dV, dsv] = mulGate.backward(V, this.s, dmulv)
		const ds = dsv + diff_s;
		const dadd = activation.backward(this.add, ds);
		const [dmulw, dmulu] = addGate.backward(this.mulw, this.mulu, dadd);
		const [dW, dprev_s] = mulGate.backward(W, prev_s, dmulw)
        const [dU, dX] = mulGate.backward(U, X, dmulu)
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
	this.initialize = (word_dim, hidden_dim=100, bptt_truncate=4) => {
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
        for(let i=0;i<hidden_dim;i++){
        	this.V.push([]);
        	for(let j=0;j<hidden_dim;j++){
        		const r = (Math.random() * Math.sqrt(1/word_dim) * 2) - Math.sqrt(1/hidden_dim);
        		this.V[i].push(r)
        	}
        }
	}
	this.forward_propagation = (X) => {
		//Here X is a sentence.
		let total_t = X.length;
		let layers = [];
		let prev_s = math.zeros(this.hidden_dim);
		for(let t=0;t<total_t;t++){
			const layer = new RNNLayer();
			let input = math.zeros(this.hidden_dim);
			input[x[t]] = 1;
			layer.forward(input, prev_s, this.U, this.W, this.V)
			prev_s = layer.
			layers.push(layer)
		}
		return layers
	}
	this.predict = (X) => {
		const output = new Softmax();
		const layers = this.forward_propagation(x);
		return layers.map((layer)=>{
			return argMax(output.predict(layer.mulv));
		})
	}
	this.calculate_loss = (X, Y) => {
		const output = new Softmax();
		const layers = this.forward_propagation(x);
		let loss = 0;
		layers.forEach((layer,index) => {
			loss += output.loss(layer.mulv, Y[index])
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
		const output = Softmax();
    	const layers = this.forward_propagation(x);
    	const total_t = layers.length;
    	let prev_s_t = math.zeros(this.hidden_dim)
    	let diff_s = math.zeros(this.hidden_dim)
    	let dmulv;
    	let input;
    	let dprev_s, dU_t, dW_t, dV_t, dU_i, dW_i, dV_i;
    	let dV = math.zeros(this.V.length, this.V[0].length)
    	let dU = math.zeros(this.U.length, this.U[0].length)
    	let dW = math.zeros(this.W.length, this.W[0].length)

    	for(let t = 0 ; t<total_t;t++){
    		dmulv = output.diff(layers[t].mulv, y[t])
        	input = math.zeros(this.word_dim)
        	input[x[t]] = 1
        	[dprev_s, dU_t, dW_t, dV_t] = layers[t].backward(input, prev_s_t, this.U, this.W, this.V, diff_s, dmulv)
        	prev_s_t = layers[t].s
        	dmulv = math.zeros(this.word_dim)
    		for(let i=t-1;i>Math.max(-1, t-this.bptt_truncate-1);i--){
    			input = math.zeros(this.word_dim)
    			input[x[i]] = 1
    			const prev_s_i = (i == 0) ? math.zeros(this.hidden_dim) : layers[i-1].s
        	    [dprev_s, dU_i, dW_i, dV_i] = layers[i].backward(input, prev_s_i, this.U, this.W, this.V, dprev_s, dmulv)
        	    dU_t = math.add(dU_i, dU_t)
        	    dW_t = math.add(dW_i, dW_t)
    		}
    		dV = math.add(dV_t, dV)
        	dU = math.add(dU_t, dU)
        	dW = math.add(dW_t, dW)
    	}
    	return [dU, dW, dV];
	}
	this.sgd_step = (x, y, learning_rate) => {
		let [dU, dW, dV] = this.bptt(x, y)
		this.U -= learning_rate * dU
    	this.V -= learning_rate * dV
    	this.W -= learning_rate * dW
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

