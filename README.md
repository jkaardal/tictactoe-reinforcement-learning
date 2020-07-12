# "Deep" reinforcement learning of Tic-Tac-Toe

This package trains an agent with an approximate policy using a dense neural network of arbitrary depth. Residual
connections are provided at defined intervals to avoid issues like exploding/disappearing gradients. This is all
implemented in tensorflow >= 2.0 using eager execution.

Three policy approximation techniques are provided:
- REINFORCE,
- REINFORCE with baseline,
- and actor-critic.

An approximate state-value function is used for the baseline in the latter two policies. Two types of value functions
are provided:
- tabular
- and quadratic polynomial approximation.

Both are updated using a Monte Carlo algorithm.

For a guide to the parameters to train this model, see the help string to the `train.py` module:
```sh
python -m tictactoe-reinforcement-learning.train -h
```
which is the main entrypoint.

For more thorough background information and a summary of the theory, please read 
[this link](http://joelkaardal.com/links/tutorials/tictactoe.html).

For a playable demo of an entirely tabular implementation in javascript, see 
[this link](http://joelkaardal.com/links/demos/xo-rl-js/xo.html) with source code found in
[this repository](https://github.com/jkaardal/xo-rl-js).