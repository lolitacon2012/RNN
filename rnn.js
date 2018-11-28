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
const generated_training_set = corpus.generate(10000);
const fake_set = [
"nobe chikaku / iei shisereba / uguisu no / naku naru koe wa / asa na asa na kiku / "
,"kasuga no wa / kyo~ wa na yaki so / wakakusa no / tsuma mo komoreri / ware mo komoreri / "
,"kasuga no no / tobuhi no nomori / idete miyo / ima ikuka arite / wakana tsumite / "
,"miyama ni wa / matsu no yuki dani / kienaku ni / miyako wa nobe no / wakana tsumikeri / "
,"azusayumi / oshite harusame / kyo~ furinu / asu sae furaba / wakana tsumiten / "
,"kimi ga tame / haru no no ni idete / wakana tsumu / waga koromode ni / yuki wa furitsutsu / "
,"kasuga no no / wakana tsumi ni ya / shirotae no / sode furihaete / hito no yuku ran / "
,"haru no kuru / kasumi no koromo / nuki o usumi / yamakaze ni koso / midaru beranare / "
,"tokiwa naru / matsu no midori mo / haru kureba / ima hitoshio no / iro masarikeri / "
,"waga seko ga / koromo haru same / furu goto ni / nobe no midori zo / iro masarikeru / "
,"aoyagi no / ito yorikakuru / haru shi mo zo / midarete hana no / hokorobinikeru / "
,"asamidori / ito yorikakete / shiratsuyu o / tama ni mo nukeru / haru no yanagi ka / "
,"momochidori / saezuru haru wa / mono goto ni / aratamaredomo / ware zo furiyuku / "
,"ochikochi no / tatsuki mo shiranu / yama naka ni / obotsukanaku mo / yobukodori kana / "
,"haru kureba / kari kaeru nari / shiragumo no / michi yukiburi ni / koto ya tsute mashi / "
,"harugasumi / tatsu o misutete / yuku kari wa / hana naki sato ni / sumi ya naraeru / "
,"oritsureba / sode koso nioe / ume no hana / ari to ya koko ni / uguisu no naku / "
,"iro yori mo / ka koso aware to / omo~yure / ta ga sode fureshi / yado no ume zo mo / "
,"yado chikaku / ume no hana ueji / ajikinaku / natsu hito no ka ni / ayamatarekeri / "
,"ume no hana / tachi yoru bakari / arishi yori / hito no no togamuru / ka ni zo shiminuru / "
,"uguisu no / kasa ni nuu cho~ / ume no hana / orite kazasan / oi kakuru ya to / "
,"yoso ni nomi / aware to zo mishi / ume no hana / akanu iro ka wa / orite narikeri / "
,"kimi narade / tare ni ka misen / ume no hana / iro o mo ka o mo / shiru hito zo shiru / "
,"ume no hana / niou harube wa / kurabu yama / yami ni koyuredo / shiruku zo arikeru / "
,"tsuki yo ni wa / sore to mo miezu / ume no hana / ka o tazunete zo / shirubekarikeru / "
,"haru no yo no / yami wa ayanashi / ume no hana / iro koso miene / ka ya wa kakururu / "
,"hito wa isa / kokoro mo shirazu / furusato wa / hana zo mukashi no / ka ni nioikeru / "
,"haru goto ni / nagaruru kawa o / hana to mite / orarenu mizu ni / sode ya nurenan / "
,"toshi o hete / hana no kagami to / naru mizu wa / chirikakaru o ya / kumoru to iu ran / "
,"kuru to aku to / me karenu mono o / ume no hana / itsu no hitoma ni / utsuroinuran / "
,"ume ga ka o / sode ni utsushite / todometeba / haru wa sugu to mo / katami naramashi / "
,"chiru to mite / aru beki mono o / ume no hana / utate nioi no / sode ni tomareru / "
,"chirinu tomo / ka o dani nokose / ume no hana / koishiki toki no / omoide ni sen / "
,"kotoshi yori / haru shirisomuru / sakurabana / chiru to iu koto wa / narawazaranan / "
,"yama takami / hito mo susamenu / sakurabana / itaku na wabi so / ware mihayasan / "
,"yamazakura / waga mi ni kureba / harugasumi / mine ni mo o ni mo / tachikakushitsutsu / "
,"toshi fureba / yowai wa oinu / shika wa aredo / hana o shi mireba / mono omoi mo nashi / "
,"yo no naka ni / taete sakura no / nakariseba / haru no kokoro wa / nodokekaramashi / "
,"ishi hashiru / taki naku mo gana / sakurabana / taorite mo kon / minu hito no tame / "
,"mite nomi ya / hito ni kataran / sakurabana / te goto ni orite / iezu to ni sen / "
,"miwataseba / yanagi sakura o / kokimazete / miyako zo haru no / nishiki narikeru / "
,"iro mo ka mo / onaji mukashi ni / sakuramedo / toshi furu hito zo / aratamarikeru / "
,"tare shi ka mo / tomete oritsuru / harugasumi / tachikakusu ran / yama no sakura o / "
,"sakurabana / sakinikerashi na / ashihiki no / yama no kai yori / miyuru shirakumo / "
,"miyoshino no / yamabe ni sakeru / sakurabana / yuki ka to nomi zo / ayamatarekeru / "
,"sakurabana / haru kuwawareru / toshi dani mo / hito no kokoro ni / akare ya wa senu / "
,"adanari to / na ni koso tatere / sakurabana / toshi ni mare naru / hito mo machikeri / "
,"kyo~ kozuba / asu wa yuki to zo / furinamashi / kiezu wa aritomo / hana to mimashi ya / "
,"chirinureba / kouredo shirushi / naki mono o / kyo~ koso sakura / oraba oriteme / "
,"oritoraba / oshige ni mo aru ka / sakurabana / iza yado karite / chiru made wa min / "
,"sakura iro ni / koromo wa fukaku / somete kin / hana no chirinan / nochi no katami ni / "
,"waga yado no / hanami gatera ni / kuru hito wa / chirinan nochi zo / koishikaru beki / "
,"miru hito mo / naki yamazato no / sakurabana / hoka no chirinan / nochi zo sakamashi / "
,"harugasumi / tanabiku yama no / sakurabana / utsurowan to ya / iro kawari yuku / "
,"mate to iu ni / chirade shi tomaru / mono naraba / nani o sakura ni / omoimasamashi / "
,"nokorinaku / chiru zo medetaki / sakurabana / arite yo no naka / hate no ukereba / "
,"kono sato ni / tabine shinu beshi / sakurabana / chiri no magai ni / ieji wasurete / "
,"utsusemi no / yo ni mo nitaru ka / hanazakura / saku to mishi ma ni / katsu chirinikeri / "
,"sakurabana / chiraba chiranan / chirazu tote / furusatobito no / kite mo minaku ni / "
,"sakura chiru / hana no tokoro wa / haru nagara / yuki zo furitsutsu / kiegate ni suru / "
,"hana chirasu / kaze no yadori wa / tare ka shiru / ware ni oshieyo / yukite uramin / "
,"iza sakura / ware mo chirinan / hito sakari / arinaba hito ni / ukime mienan / "
,"hitome mishi / kimi mo ya kuru to / sakurabana / kyo~ wa machimite / chiraba chiranan / "
,"harugasumi / nani kakusu ran / sakurabana / chiru ma o dani mo / miru beki mono o / "
,"tarekomete / haru no yukue no / shiranu ma ni / machishi sakura mo / utsuroinikeri / "
,"eda yori mo / ada ni chirinishi / hana nareba / ochite mo mizu no / awa to koso nare / "
,"koto naraba / sakazu ya wa aranu / sakurabana / miru ware sae ni / shizugokoro nashi / "
,"sakurabana / toku chirinu tomo / omo~ezu / hito no kokoro zo / kaze mo fukiaenu / "
,"hisakata no / hikari nodokeki / haru no hi ni / shizugokoro naku / hana no chiru ran / "
,"haru kaze wa / hana no atari o / yogite fuke / kokorozu kara ya / utsurou to min / "
,"yuki to nomi / furu dani aru o / sakurabana / ika ni chire to ka / kaze no fuku ran / "
,"yama takami / mitsutsu waga koshi / sakurabana / kaze wa kokoro ni / makasu beranari / "
,"harusame no / furu wa namida ka / sakurabana / chiru o oshimanu / hito shi nakereba / "
,"sakurabana / chirinuru kaze no / nagori ni wa / mizu naki sora ni / nami zo tachikeru / "
,"furusato to / narinishi nara no / miyako ni mo / iro wa kawarazu / hana wa sakikeri / "
,"hana no iro wa / kasumi ni komete / misezu tomo / ka o dani nusume / haru no yama kaze / "
,"hana no ki mo / ima wa horiueji / haru tateba / utsurou iro ni / hito naraikeri / "
,"haru no iro no / itari itaranu / sato wa araji / sakeru sakazaru / hana no miyuran / "
,"miwa yama o / shikamo kakusu ka / harugasumi / hito ni shirarenu / hana ya sakuran / "
,"iza kyo~ wa / haru no yamabe ni / majiri nan / kurenaba nage no / hana no kage ka wa / "
,"itsu made ka / nobe ni kokoro no / akugaren / hana shi chirazuba / chiyo mo henu beshi / "
,"haru goto ni / hana no sakari wa / arinamedo / aimin koto wa / inochi narikeri / "
,"hana no goto / yo no tsune naraba / sugushite shi / mukashi wa mata mo / kaerikinamashi / "
,"fuku kaze ni / atsuraetsukuru / mono naraba / kono hito moto wa / yogiyo to iwamashi / "
,"matsu hito mo / konu mono yue ni / uguisu no / nakitsuru hana o / oritekeru kana / "
,"saku hana wa / chigusu nagara ni / ada naredo / tare ka wa haru o / uramihatetaru / "
,"harugasumi / iro no chigusa ni / mietsuru wa / tanabiku yama no / hana no kage kamo / "
,"kasumi tatsu / haru no yamabe wa / tokeredo / fukikuru kaze wa / hana no ka zo suru / "
,"hana mireba / kokoro sae ni zo / utsurikeru / iro ni wa ideji / hito mo koso shire / "
,"uguisu no / naku nobe goto ni / kite mireba / utsurou hana ni / kaze zo fukikeru / "
,"fuku kaze o / nakite urami yo / uguisu wa / ware ya wa hana ni / te dani furetaru / "
,"chiru hana no / naku ni shi tomaru / mono naraba / ware uguisu ni / otoramashi ya wa / "
,"hana no chiru / koto ya wabishiki / harugasumi / tatsuta no yama no / uguisu no koe / "
,"kozutaeba / ono ga ha kaze ni / chiru hana o / tare ni o~sete / kokora nakuran / "
,"shirushi naki / ne o mo naku kana / uguisu no / kotoshi nomi chiru / hana naranaku ni / "
,"koma namete / iza mi ni yukan / furusato wa / yuki to nomi koso / hana wa chirurame / "
,"chiru hana o / nani ka uramin / yo no naka ni / waga mi mo tomo ni / aran mono ka wa / "
,"hana no iro wa / utsurinikeri na / itazura ni / waga mi yo ni furu / nagame seshi ma ni / "
,"oshi to omou / kokoro wa ito ni / yorarenan / chiru hana goto ni / nukite todomen / "
,"azusayumi / haru no yamabe o / koe kureba / michi mo sariaezu / hana zo chirikeru / "
,"haru no no ni / wakana tsuman to / koshi mono o / chirikau hana ni / michi wa madoinu / "
,"yadori shite / haru no yamabe ni / netaru yo wa / yume no uchi ni mo / hana zo chirikeru / "
,"fuku kaze to / tani no mizu to shi / nakariseba / miyamagakure no / hana o mimashi ya / "
,"yoso ni mite / kaeran hito ni / fuji no hana / haimatsuware yo / eda wa oru tomo / "
,"waga yado ni / sakeru fujinami / tachikaeri / sugikate ni nomi / hito no miru ran / "
,"ima mo kamo / sakiniou ran / tachibana no / kojima no saki no / yamabuki no hana / "
,"harusame ni / nioeru iro mo / akanaku ni / ka sae natsukashi / yamabuki no hana / "
,"yamabuki wa / ayana na saki so / hana min to / ueken kimi ga / koyoi konaku ni / "
,"yoshino gawa / kishi no yamabuki / fuku kaze ni / soko no kage sae / utsuroinikeri / "
,"kawazu naku / ide no yamabuki / chirinikeri / hana no sakari ni / awamashi mono o / "
,"omoudochi / haru no yamabe ni / uchimurete / soko to mo iwanu / tabine shite shi ga / "
,"azusayumi / haru tachishi yori / toshi tsuki no / iru ga gotoku mo / omo~yuru kana / "
,"naki tomuru / hana shi nakereba / uguisu mo / hate wa monouku / narinu beranari / "
,"hana chireru / mizu no manimani / tomekureba / yama ni wa haru mo / nakunarinikeri / "
,"oshimedomo / todomaranakuni / harugasumi / kaeru michi ni shi / tachinu to omoeba / "
,"koe taezu / nake ya uguisu / hitotose ni / futatabi to dani / kubeki haru ka wa / "
,"todomu beki / mono to wa nashi ni / hakanaku mo / chiru hana goto ni / tagu~ kokoro ka / "
,"nuretsutsu zo / shiite oritsuru / toshi no uchi ni / haru wa iku ka mo / araji to omoeba / "
,"kyo~ nomi to / haru o omowanu / toki dani mo / tatsu koto yasuki / hana no kage ka wa / "
,"waga yado no / ike no fujinami / sakinikeri / yama hototogisu / itsu ka kinakan / "
,"aware cho~ / koto o amata ni / yaraji to ya / haru no okurete / hitori sakuran / "
,"satsuki matsu / yama hototogisu / uchi habuki / ima mo nakanan / kozo no furugoe / "
,"satsuki koba / naki mo furinan / hototogisu / madashiki hodo no / koe o kikabaya / "
,"satsuki matsu / hana tachibana no / kao kageba / mukashi no hito no / sode no ka zo suru / "
,"itsu no ma ni / satsuki kinu ran / ashihiki no / yama hototogisu / ima zo naku naru / "
,"kesa kinaki / imada tabi naru / hototogisu / hanatachibana ni / yado wa karanan / "
,"otowa yama / kesa koekureba / hototogisu / kozue karuka ni / ima zo naku naru / "
,"hototogisu / hatsukoe kikeba / ajikinaku / nushi sadamaranu / koe seraru hata / "
,"isonokami / furuki miyako no / hototogisu / koe bakari koso / mukashi narikere / "
,"natsu yama ni / naku hototogisu / kokoro araba / mono omou ware ni / koe na kikase so / "
,"hototogisu / naku koe kikeba / wakarenishi / furusato sae zo / koishikarikeru / "
,"hototogisu / na ga naku sato no / amata areba / nao utomarenu / omou mono kara / "
,"omoiizuru / tokiwa no yama no / hototogisu / karakurenai no / furiidete zo naku / "
,"koe wa shite / namida wa mienu / hototogisu / waga koromode no / hitsu o karanan / "
,"ashihiki no / yama hototogisu / orihaete / tare ka masaru to / ne o nomi zo naku / "
,"ima sara ni / yama e kaeru na / hototogisu / koe no kagiri wa / waga yado ni nake / "
,"ya yo ya mate / yama hototogisu / kotozuten / ware yo no naka ni / sumiwabinu to yo / "
,"samidare ni / mono omoi oreba / hototogisu / yobukaku nakite / izuchi yuku ran / "
,"yo ya kuraki / michi ya madoeru / hototogisu / waga yado o shi mo / sugikate ni naku / "
,"yadori seshi / hana tachibana mo / karenaku ni / nado hototogisu / koe taenu ran / "
,"natsu no yo no / fusu ka to sureba / hototogisu / naku hitokoe ni / akuru shinonome / "
,"kururu ka to / mireba akenuru / natsu no yo o / akazu to ya naku / yama hototogisu / "
,"natsu yama ni / koishiki hito ya / kiriniken / koe furitatete / naku hototogisu / "
,"kozo no natsu / naki furushiteshi / hototogisu / sore ka aranu ka / koe no kawaranu / "
,"samidare no / sora mo todoro ni / hototogisu / nani o ushi to ka / yo tadanaku ran / "
,"hototogisu / koe mo kikoezu / yamabiko wa / hoka ni naku ne o / kotae ya wa senu / "
,"hototogisu / hito matsu yama ni / naku nareba / ware uchitsuke ni / koi masarikeri / "
,"mukashihe ya / ima mo koishiki / hototogisu / furusato ni shi mo / nakite kitsuran / "
,"hototogisu / ware to wa nashi ni / unohana no / uki yo no naka ni / nakiwataru ran / "
,"hachisuba no / nigori ni shimanu / kokoro mote / nani ka wa tsuyu o / tama to azamuku / "
,"natsu no yo wa / mada yoi nagara / akenuru o / kumo no izuko ni / tsuki yadoru ran / "
,"chiri o dani / sueji to zo omou / sakishi yori / imo to waga nuru / tokonatsu no hana / "
,"natsu to aki to / yukikau sora no / kayoiji wa / katae suzushiki / kaze ya fuku ran / "
,"aki kinu to / me ni wa sayaka ni / mienedomo / kaze no oto ni zo / odorokarenuru / "
,"kawa kaze no / suzushiku mo aru ka / uchiyosuru / nami to tomo ni ya / aki wa tatsu ran / "
,"waga seko ga / koromo no suso o / fukikaeshi / uramezurashiki / aki no hatsukaze / "
,"kino~ koso / sanae torishika / itsu no ma ni / inaba soyogite / akikaze no fuku / "
,"aki kaze no / fukinishi hi yori / hisakata no / ama no kawara ni / tatanu hi wa nashi / "
,"hisakata no / ama no kawara no / watashimori / kimi watarinaba / kaji kakushite yo / "
,"ama no kawa / momiji o hashi ni / watasebaya / tanabata tsu me no / aki o shi mo matsu / "
,"koikoite / au yo wa koyoi / ama no kawa / kiri tachiwatari / akezu mo aranan / "
,"ama no kawa / asase shiranami / tadoritsutsu / watarihateneba / ake zo shinikeru / "
,"chigiriken / kokoro zo tsuraki / tanabata no / toshi ni hitotabi / au wa au ka wa / "
,"toshi goto ni / au to wa suredo / tanabata no / neru yo no kazu zo / sukunakarikeru / "
,"tanabata ni / kashitsuru ito no / uchihaete / toshi no o nagaku / koi ya wataran / "
,"koyoi kon / hito ni wa awaji / tanabata no / hisashiki hodo ni / machi mo koso sure / "
,"ima wa to te / wakaruru toki wa / ama no kawa / wataranu saki ni / sode zo hichinuru / "
,"kyo~ yori wa / ima kon toshi no / kino~ o zo / itsu shika to nomi / machiwatarubeki / "
,"ko no ma yori / morikuru tsuki no / kage mireba / kokorozukushi no / aki wa kinikeri / "
,"o~kata no / aki kuru kara ni / waga mi koso / kanashiki mono to / omoishirinure / "
,"waga tame ni / kuru aki ni shimo / aranaku ni / mushi no ne kikeba / mazu zo kanashiki / "
]
//fake_set above
const super_fake = ["Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."]

fake_set.join('\n').split(" ").map((s)=>{
	s.replace(/[.,<>/?!@"'{}\n;:=+\-_)(*&^%$@)]/g, (a)=>{
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
const hidden_dim = 31;
const word_dim = indexToWord.size;
const rnn = new Model();
console.log(`
	word size = ${word_dim}`)
rnn.initialize(word_dim, hidden_dim);
rnn.train(X_train,Y_train,0.8,7,1);

const to_predict = [0, ...fake_set[0].split(" ").map((a)=>{
	return wordToIndex.get(a)
})]
const generate_length_max = 1000;
let output_full = [0];
for(let m=0;m<generate_length_max;m++){
	const step = rnn.predict([output_full[m]]);
	output_full.push(step[0]);
}
console.log("OUTPUT = " + output_full.map((e)=>{
	return indexToWord.get(e);
}).slice(1).join(" "))


