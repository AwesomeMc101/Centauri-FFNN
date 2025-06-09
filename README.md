# Tensora-FFNN

## Full C++ Feed-Forward Neural Network Implementation (No Libraries)

Yep, I'm 17 and wrote this in 3.5 hours. And they say some programmers write ~200 lines/month at REAL jobs!!!
Just lock in fr.

# Training Context
* Network is set up to learn basic addition. Data is generated randomly in the `sums` function.

# Network Class
* This class contains each layer + the MSE loss class. The constructor has 1 mandatory input alongside two options:
* - InputData _id -> Class containing training data
  - BOOL _doAutoLink -> Autolink layers
  - BOOL _doInternalChecks -> Should the layer generation wrapper enable layer internal safety checks?
* `link()` -> Links layers together. "omg its so complex why would you make it like that!!" no you just don't get pointers fr.
  - READER_FORWARD -> vector to read from during forward pass
  - READER_ACTIVATION -> front layer's activation for backprop (perform RELU derivative?)
  - READER_BACKWARD -> front layer's deriv input
  - MSE_T_READER -> MSE read from
  - MSE_T_YHAT -> MSE actual outputs for loss calc
* `new_layer(inputs, neurons)` -> Create new layer.
* `execute(epochs, learning_rate)` -> Train.
* `output_learned_op()` -> debug function to view outputs from training data.
* `predict(val1, val2)` -> embedded predict built on the training context    

# Iteration Flow
* Iterates through each layers in normal FFNN fashion.
## Layer Forward Pass
### Internal Checking
* Ensures the layer is valid to run by checking for null reader pointers and ensuring weights/biases & activation are initialized.
### Arith
* Read from past layer's outputs OR training data inputs.
* DOT [input * t_weights] + bias
* RELU ^^

## MSE F/B
* Forward pass calculates loss obv.
* Backward pass is negative bc subtract and calc dinputs

## Layer Backward Pass
* Iterate in reverse.
* Backpropagate ReLU.
* Calculate dinputs = DOT [forward_dinputs * weights]
* Calculate dweights = DOT [t_input * t_dvalues]
* ### Important Note -> if training twice on different sizes of data, change the rs input on the DOT functions to TRUE. This way the dinputs/weights will be automatically resized.
* Dbiases are accumulated t_dvalues.
## Stochastic GD
* Extremely basic. No AMSGrad/Adam and no momentum it works as basic as possible.

# Thanks for using! Remember that there's AI majors who couldn't code this ;(
