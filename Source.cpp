/*
* 
** Source.cpp (FFNN Proto/Centauri-FFNN)
** Registered under MIT
*
** Written in whole by AwesomeMc101 / C.D.H
*
** Refer to the github README.md for details.
* 
*/

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>


#ifdef WINDOWS
#include <Windows.h>
#else
typedef unsigned short USHORT;
typedef unsigned int UINT;
typedef bool BOOL;
#define TRUE true
#define FALSE false
#endif

#define CRASH_ON_ERROR

typedef long double NINT;

namespace ErrorManager {
	typedef enum {
		NIL_SIZE,
		WRONG_SIZE,
		NO_ACTIVATION,
		WRONG_READER_TYPE,
		INVALID_LAYER_MULTS, //ig?? ;(
		INVALID_LAYER_READERS,
		INVALID_LAYER_ACTIVATION,
		LAYER_INTERNAL_CHECK_FAILURE,
		NO_POINTER_CODE,
		NO_LINK_MSECOUNTER,
		WRONG_READER_TYPE_MSE,
		MEAN_NOSIZE,
		NORMALIZER_NOGLOBAL,
		NORMALIZER_NOGLOBAL_REVERSION,
	} ErrorCode;

	void pushError(ErrorCode c) {
		std::cout << "error " << c << " pushed.\n";
#ifdef CRASH_ON_ERROR
		abort();
#endif
	}
}
using ErrorManager::ErrorCode;
using ErrorManager::pushError;

/* return codes */
#define FAIL 0
#define SUCCESS 1
typedef bool REF_RETURN; //return value for functions which perform modifications on references/pointers



namespace Arith {
	REF_RETURN transpose(std::vector<std::vector<NINT>>& r) {
		if (!r.size() || !r[0].size()) { //-> r.size()==0, autothrow if no check r[0]
			pushError(ErrorCode::NIL_SIZE);
			return FAIL; 
		}
		std::vector<std::vector<NINT>> x(r[0].size(), std::vector<NINT>(r.size(), 0));
		for (int i = 0; i < r.size(); i++) {
			for (int j = 0; j < r[0].size(); j++) {
				x[j][i] = r[i][j];
			}
		}
		r = x; //either way we copy sm btw
		return SUCCESS;
	}

	REF_RETURN dot(std::vector<std::vector<NINT>>& base, const std::vector<std::vector<NINT>>& sec) {
		if (!base.size() || !base[0].size() || !sec.size() || !sec[0].size()) {
			pushError(ErrorCode::NIL_SIZE);
			return FAIL;
		}

		std::vector<std::vector<NINT>> output(base.size(), std::vector<NINT>(sec.size(), 0));
		for (int i = 0; i < base.size(); ++i) {
			for (int j = 0; j < sec.size(); ++j) {
				output[i][j] = std::inner_product(base[i].begin(), base[i].end(), sec[j].begin(), static_cast<NINT>(0));
			}
		}
		base = output;
		return SUCCESS;
	}
	REF_RETURN dot(std::vector<std::vector<NINT>>& output, const std::vector<std::vector<NINT>>& base, const std::vector<std::vector<NINT>>& sec, BOOL rs = TRUE) {
		if (!base.size() || !base[0].size() || !sec.size() || !sec[0].size()) {
			pushError(ErrorCode::NIL_SIZE);
			return FAIL;
		}

		if (rs) {//bro what
			output = std::vector<std::vector<NINT>>(base.size(), std::vector<NINT>(sec.size(), 0));
		}

		for (int i = 0; i < base.size(); i++) {
			for (int j = 0; j < sec.size(); j++) {
				output[i][j] = std::inner_product(base[i].begin(), base[i].end(), sec[j].begin(), static_cast<NINT>(0));
			}
		}
		return SUCCESS;
	}

	NINT mean(const std::vector<NINT>& r) {
		if (r.empty()) { pushError(ErrorCode::MEAN_NOSIZE); return 0; }
		return (std::accumulate(r.begin(), r.end(), static_cast<NINT>(0)) / r.size());
	}
}

class InputData {
public:
	std::vector<std::vector<NINT>> input;
	std::vector<NINT> output;
};

namespace Normalizer {
	//laziness peak
	static NINT g_n1 = 0; //Mean (id input0)
	static NINT g_n2 = 0; //Mean (id input1)
	static NINT g_n3 = 0; //Mean (id output)

	void init_global(const InputData& id) {
		auto transposed_input = id.input;
		Arith::transpose(transposed_input);

		g_n1 = Arith::mean(transposed_input[0]);
		g_n2 = Arith::mean(transposed_input[1]);
		g_n3 = Arith::mean(id.output);
	}

	void normalize_set(InputData& id) {
		if (!g_n1 || !g_n2 || !g_n3) {
			pushError(ErrorCode::NORMALIZER_NOGLOBAL);
			printf("Automatically calculating globals.\n");
			init_global(id);
		}

		for (int i = 0; i < id.input.size(); i++) {
			id.input[i][0] /= g_n1;
			id.input[i][1] /= g_n2;
			id.output[i] /= g_n3;
		}
	}
	void normalize_mset(std::vector<std::vector<NINT>>& s) {
		if (!g_n1 || !g_n2 || !g_n3) {
			pushError(ErrorCode::NORMALIZER_NOGLOBAL);
			abort();
		}

		for (int i = 0; i < s.size(); i++) {
			s[i][0] /= g_n1;
			s[i][1] /= g_n2;
		}
	}
	NINT revert_output(NINT r) {
		if (!g_n1 || !g_n2 || !g_n3) {
			pushError(ErrorCode::NORMALIZER_NOGLOBAL_REVERSION);
			return r;
		}
		return (r * g_n3);
	}
	
}

typedef enum {
	RELU,
	SOFTMAX,
	BINCLASS,
	LINEAR, //this may be ez to add ;d

	NOACT
} Activation;
namespace NN_Activation {

	void perform_relu(NINT& v) {
		v = (v < 0 ? 0 : v); //let me ve4rify this YES lol
	}
	void perform_backward_relu(NINT& v) {
		v = (v < 0 ? 0 : 1);
	}
}
#define isValidActivation(A) ((int)A < (int)NOACT)

//mean squared error is ALL IM ADDING :D :D :D :D :D :D 
class MSELayer {
private:
	//i was GONNA be lazy but ill do it properly
	std::vector<std::vector<NINT>>* reader;
	std::vector<NINT>* yt;

	std::vector<std::vector<NINT>> dinputs;
public:
	NINT loss = 10e2;
	std::vector<std::vector<NINT>>* dinp_ptr() {
		return &dinputs;
	}

#define MSE_T_READER 1
#define MSE_T_YHAT 2
	/* omg omg you called it yhat omg omg omg no way*/
	REF_RETURN attach_reader(UINT code, void* ptr) {
		switch (code) {
		case MSE_T_READER:
			reader = static_cast<std::vector<std::vector<NINT>>*>(ptr);
			return SUCCESS;
		case MSE_T_YHAT:
			yt = static_cast<std::vector<NINT>*>(ptr);
			return SUCCESS;
		}
		pushError(ErrorCode::WRONG_READER_TYPE_MSE);
		return FAIL;
	}


	void forward() {
		loss = 0;

		for (int i = 0; i < yt->size(); i++) {
			NINT ty = (((*yt)[i])); //true
			NINT py = (((*reader)[i][0])); //prediction
			loss += ((ty - py) * (ty - py));
		}
		loss /= (yt->size() > 0 ? yt->size() : 1);
	}
	void backward() {
		dinputs.clear();
		dinputs.reserve(yt->size());
		for (int i = 0; i < yt->size(); i++) {
			dinputs.emplace_back(
				std::vector<NINT>(1,
					-2 * /* i know the derivative is 2x but its cuz we add later IK THE PRODUCT RULE */
					((*yt)[i] - (*reader)[i][0])
					/ yt->size()
					)
			);
			
		}
	//	std::cout << "Dinput addr: " << &dinputs << "\n";
	}
};

class DeepLayer {
private:
	std::vector<std::vector<NINT>> weights;
	std::vector<NINT> biases;

	std::vector<std::vector<NINT>> dweights;
	std::vector<std::vector<NINT>> dinputs;
	std::vector<NINT> dbiases;

	std::vector<std::vector<NINT>>* read_f, * read_b;
	std::vector<std::vector<NINT>> input, output; //provided read_f/read_b

	Activation A = Activation::NOACT;
	Activation* pA = nullptr;
	BOOL internal_checking;
public:

	//you might be asking why i dont just move these to public.... dont ask that again.
#define IP_DINPUT 0
#define IP_OUTPUT 1
#define IP_ACT 2
	void* get_internal_pointer(USHORT code) {
		switch (code) {
		case IP_DINPUT:
			return &dinputs;
		case IP_OUTPUT:
			return &output;
		case IP_ACT:
			return &A;
		}
		pushError(ErrorCode::NO_POINTER_CODE);
		return 0x0000;
	}

	REF_RETURN verify_layer() {
		//had to verify this obscure method
		//let it push every error necessary. no early rewt.

		REF_RETURN val = SUCCESS;
		if (weights.data() == nullptr ||
			biases.data() == nullptr
			) {
			pushError(ErrorCode::INVALID_LAYER_MULTS);
			val = FAIL;
		}

		if (read_f == nullptr || read_b == nullptr) {
			pushError(ErrorCode::INVALID_LAYER_READERS);
			val = FAIL;
		}

		if (!isValidActivation(A)) {
			pushError(ErrorCode::INVALID_LAYER_ACTIVATION);
			val = FAIL;
		}
		return val;
	}

	USHORT getOutputSize() {
		return (weights.size() ? weights[0].size() : 0);
	}

	DeepLayer(
		/* main settings -> */ UINT inputs, UINT neurons, Activation a = RELU,
		/* readers -> */	std::vector<std::vector<NINT>>* in = 0x0, std::vector<std::vector<NINT>>* back = 0x0,
		/* alt. settings -> */	BOOL internal_checks = 1) {
		internal_checking = internal_checks;
		if (!inputs || !neurons) {
			pushError(ErrorCode::WRONG_SIZE);
			return;
		}
		if (!isValidActivation(a)) {
			pushError(ErrorCode::NO_ACTIVATION);
			return;
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<NINT> dist(0, sqrtl(2.0 / inputs));
		weights = std::vector<std::vector<NINT>>(inputs, std::vector<NINT>(neurons));
		for (auto& row : weights) {
			for (NINT& val : row) {
				val = dist(gen);
			}
		}

		biases = std::vector<NINT>(neurons, 0);

		read_f = in;
		read_b = back;

		A = a;
	}

#define READER_FORWARD 1
#define READER_BACKWARD 2
#define READER_ACTIVATION 3
	REF_RETURN alterReader(UINT type, void* new_pointer) {
		switch (type) {
		case READER_FORWARD:
			read_f = static_cast<std::vector<std::vector<NINT>>*>(new_pointer);
			return SUCCESS;
		case READER_BACKWARD:
			read_b = static_cast<std::vector<std::vector<NINT>>*>(new_pointer);
			return SUCCESS;
		case READER_ACTIVATION:
			pA = static_cast<Activation*>(new_pointer);
			return SUCCESS;
		}

		//neither?!?
		pushError(ErrorCode::WRONG_READER_TYPE);
		return FAIL;
	}


	void forward() {
		if (internal_checking) {
			if (!verify_layer()) {
				pushError(ErrorCode::LAYER_INTERNAL_CHECK_FAILURE);
				return;
			}
		}

		input = *read_f;
		//in future should just keep 2 weights

		std::vector<std::vector<NINT>> transposed_weights = weights;
		Arith::transpose(transposed_weights);


		if (!Arith::dot(output, input, transposed_weights)) {
			abort();
		}
		//aaaaaaaaaaaaaa
		//:) make it all one function later
		for (int i = 0; i < output.size(); i++) {
			for (int j = 0; j < output[i].size(); j++) {
				output[i][j] += biases[j];
				switch (A) {
				case RELU:
					NN_Activation::perform_relu(output[i][j]);
					break;
				case LINEAR:
					break;
				}
			}
		}

	}

	inline REF_RETURN setupDerivativeVectors(UINT i_size, UINT ir_size, UINT pd_size, UINT pdr_size) {
		dinputs.resize(pd_size, std::vector<NINT>(weights.size()));
		/*verify*/dweights.resize(ir_size, std::vector<NINT>(pdr_size)); //this takes a lot of thinking
		dbiases.resize(pdr_size);
		return SUCCESS;
	}

	void backward() {
		std::vector<std::vector<NINT>> rinput = *read_b;
		switch (*pA) {
		case RELU:
			for (int i = 0; i < rinput.size(); i++) {
				for (int j = 0; j < rinput[i].size(); j++) {
					NN_Activation::perform_backward_relu(rinput[i][j]);
				}
			}
		}
		//my energy drink is wearing off

		//Assume sizes of inputs and derivative vectors remain constant during propagation. 
		//std::cout << "INPSIZE: " << input.size() << "\nRINSIZE: " << rinput.size() << "\n";
		//std::cout << "RINad: " << &rinput << "\n";
		if (dweights.data() == 0 || dbiases.data() == 0) {
			setupDerivativeVectors(input.size(), input[0].size(), rinput.size(), rinput[0].size());
		}

		//too lazy to write another transpose
		auto transposed_input = input;
		Arith::transpose(transposed_input);

		auto transposed_dvalues = rinput;
		Arith::transpose(transposed_dvalues);

		//input FALSE on dot bc we only need to resize ONCE in the setup ^^^
		Arith::dot(dinputs, rinput, weights, FALSE);
		Arith::dot(dweights, transposed_input, transposed_dvalues, FALSE);

		for (int i = 0; i < transposed_dvalues.size(); i++) {
			dbiases[i] = std::accumulate(transposed_dvalues[i].begin(), transposed_dvalues[i].end(), static_cast<NINT>(0));
		}

		//implement regularizers l8r
	}

	void internal_sgd(NINT lr) {
		for (int i = 0; i < weights.size(); i++){
			for (int j = 0; j < weights[i].size(); j++) {
				weights[i][j] -= (lr * dweights[i][j]);
				if (i == 0) {
					biases[j] -= (lr * dbiases[j]);
				}
			}
		}
	}
};


//kindof forgot to write input data



class Network {
private:
	std::vector<DeepLayer*> L_SET;
	MSELayer* loss;

	BOOL doInternalChecks;
	BOOL doAutoLink;
	InputData id;

	void link() {
		//im so rusty oml


		if (loss == nullptr) {
			printf("Loss not found. Automatically generating an MSE layer.\n");
			loss = new MSELayer();
			if (L_SET.back()->getOutputSize() != 1) {
				printf("Final deep layer contains >1 neurons. This will not work with MSE.\nLinker failed.\n");
				pushError(ErrorCode::NO_LINK_MSECOUNTER);
				return;
			}
		}

		Activation* fake_act = (Activation*)malloc(sizeof(Activation));
		*fake_act = NOACT;
		/*
		
		
		
		*/

		for (int i = 0; i < L_SET.size(); i++) {
			//this looks like a peb unlinker :)
			L_SET[i]->alterReader(READER_FORWARD, (i > 0 ? (L_SET[i - 1]->get_internal_pointer(IP_OUTPUT)) : &id.input));
			L_SET[i]->alterReader(READER_ACTIVATION, (i < (L_SET.size() - 1) ? (L_SET[i + 1]->get_internal_pointer(IP_ACT)) : fake_act));
			L_SET[i]->alterReader(READER_BACKWARD, (i < (L_SET.size() - 1) ? (L_SET[i + 1]->get_internal_pointer(IP_DINPUT)) : loss->dinp_ptr()));
		}

		loss->attach_reader(MSE_T_READER, L_SET.back()->get_internal_pointer(IP_OUTPUT));
		loss->attach_reader(MSE_T_YHAT, &id.output);
	}

public:
	Network(InputData _id, BOOL _doAutoLink = 1, BOOL _doInternalChecks = 1) {
		id = _id;
		doInternalChecks = _doInternalChecks;
		doAutoLink = _doAutoLink;
	}

	void new_layer(UINT i, UINT n, Activation a = RELU) {
		L_SET.emplace_back((new DeepLayer(i, n, a, 0, 0, doInternalChecks)));

		if (doAutoLink) {
		//checking snap
		}
	}
	void generate_loss() {
		loss = new MSELayer();
	}
	void link_network() {
		link();
	}

	NINT execute(int epochs, NINT lr) {
		for (int i = 0; i < epochs; i++) {
			auto it = L_SET.begin();
			while (it != L_SET.end()) {
				(*it)->forward();
				it++;
			}
			it--;
			loss->forward();
			std::cout << "Loss (epoch " << i << ") -> " << loss->loss << "\n";
			loss->backward();

			for (int k = L_SET.size() - 1; k >= 0; k--) {
				L_SET[k]->backward();
			}
			//i dont have decay memorized
			NINT _lr = (lr * (1.0f / (1.0f + (1e-4 * i))));
			//wait cuz i forgot a normal decay rate too LOL
			for (DeepLayer* d : L_SET) {
				d->internal_sgd(_lr);
			}
		}
		return 1;
	}

	void output_learned_op() {
		//debugging func
		std::vector<std::vector<NINT>> outp = *(static_cast<std::vector<std::vector<NINT>>*>(L_SET.back()->get_internal_pointer(IP_OUTPUT)));
		for (int i = 0; i < id.input.size(); i++) {
			std::cout << id.input[i][0] << "+" << id.input[i][1] << " = " <<
				outp[i][0] << " (" << id.output[i] << ")\n";
		}
	}

	NINT predict(NINT v1, NINT v2) {
		std::vector<std::vector < NINT >> inp{ { v1,v2} };
		Normalizer::normalize_mset(inp);

		L_SET[0]->alterReader(READER_FORWARD, &inp);

		auto it = L_SET.begin();
		while (it != L_SET.end()) {
			(*it)->forward();
			it++;
		}

		std::vector<std::vector<NINT>> outp = *static_cast<std::vector<std::vector<NINT>>*>(L_SET.back()->get_internal_pointer(IP_OUTPUT));
		return Normalizer::revert_output(outp[0][0]);
	}
};

InputData sums(int amt) {
	InputData id;

	for (int i = 0; i < amt; i++) {
		int a1 = rand() % (amt);
		int a2 = rand() % (amt);

		id.output.push_back((a1 + a2));

		id.input.push_back({ (NINT)a1, (NINT)a2 });
	}
	return id;
}

int main() {
	//lol
	srand(time(0));
	int input_count = 2;
	
	InputData id = sums(2000);

	Normalizer::init_global(id);
	Normalizer::normalize_set(id);

	std::cout << "d si: " << id.output.size() << "\n";

	Network n(id);
	n.new_layer(2, 16);
	n.new_layer(16, 1, LINEAR);
	n.link_network();
	n.execute(100, 0.01);

	

	n.output_learned_op();
	std::cout << "Prediction: " << n.predict(432, 567) << "\n";
}
