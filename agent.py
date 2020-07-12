import numpy as np
import tensorflow as tf
from scipy.special import binom
from scipy.stats import rv_discrete


class RandomPolicy:

    def __call__(self, game):
        """
        Sample an action from this random policy and make a move.

        :param game: an instance of TicTacToe.
        """
        ac = game.valid_action_coordinates()
        assert len(ac) > 0
        row, col = ac[np.random.permutation(ac.shape[0])[0], :]
        game.move(row, col)


class ApproxPolicy:

    def __init__(self, n_layers, n_units, n_skip, discount=1.0, dtype=tf.float64):
        """
        Instantiate an ApproxPolicy agent that approximats the agent's policy with a dense neural network of
        arbitrary depth.

        :param n_layers: number of hidden layers.
        :param n_units: number of units in each hidden layer.
        :param n_skip: number of layers to skip between each residual connection.
        :param discount: discount factor. Default is 1.0.
        :param dtype: tensorflow data type to use in calculations. Default is tf.float64.
        """
        self._n_layers = n_layers
        self._n_units = n_units
        self._n_skip = n_skip
        self.discount = discount
        self.dtype = dtype
        self.np_dtype = np.float32 if dtype == tf.float32 else np.float64

        # Use Xavier initialization for weights.
        initializer = tf.initializers.GlorotNormal()

        # Declare "input" layer (really the first hidden layer).
        self.input_W = tf.Variable(
            tf.cast(initializer((9, self._n_units)), self.dtype), 
            name='input_weights', 
            dtype=self.dtype,
        )
        self.input_b = tf.Variable(
            tf.random.uniform((1, self._n_units), dtype=self.dtype), 
            name='input_bias', 
            dtype=self.dtype,
        )
 
        # Declare all requested additional hidden layers.
        self.hidden_W = []
        self.hidden_b = []
        for i in range(self._n_layers):
            self.hidden_W.append(tf.Variable(
                tf.cast(initializer((self._n_units, self._n_units)), self.dtype),
                name=f'hidden_weights_{i}',
                dtype=self.dtype,
            ))
            self.hidden_b.append(tf.Variable(
                tf.random.uniform((1, self._n_units), dtype=self.dtype),
                name=f'hidden_bias_{i}',
                dtype=self.dtype,
            ))

        # Declare the output layer.
        self.output_W = tf.Variable(
            tf.cast(initializer((self._n_units, 9)), self.dtype), 
            name='output_weights', 
            dtype=self.dtype,
        )
        self.output_b = tf.Variable(
            tf.random.uniform((1, 9), dtype=self.dtype), 
            name='output_bias', 
            dtype=self.dtype,
        )

    @property
    def variables(self):
        """
        Get a list of all weights and biases in the model.

        :return: list of bias tensors and then weight tensors in order from input to output.
        """
        biases = [self.input_b] + self.hidden_b + [self.output_b]
        weights = [self.input_W] + self.hidden_W + [self.output_W]
        return biases + weights
        
    @tf.function
    def model(self, state):
        """
        Given the current state, compute the policy over actions (\pi(a|s)).

        :param state: a nine element numpy array or tensor where 1 corresponds to 'X', -1 corresponds to 'O', and 0
            corresponds to an empty space on the game board.
        :return: a numpy array with shape (-1, 9) of action probabilities.
        """
        # Unravel state into a matrix of (samples, 9).
        state_vec = tf.reshape(tf.cast(state, self.dtype), (-1, 9))

        # Apply "input" layer.
        x = tf.matmul(state_vec, self.input_W) + self.input_b
        x = tf.nn.leaky_relu(x)

        if self._n_layers > 0:
            # Apply hidden layers with skip connections.
            skip = 1
            layers = [x]
            for i in range(self._n_layers):
                h = tf.matmul(layers[-1], self.hidden_W[i]) + self.hidden_b[i]
                if self._n_skip != 0 and skip % self._n_skip == 0:
                    h = tf.add(h, layers[-self._n_skip])
                h = tf.nn.leaky_relu(h)
                layers.append(h)
                skip += 1
            x = layers[-1]

        # Apply "output" layer.
        x = tf.matmul(x, self.output_W) + self.output_b
        # Filter out invalid actions.
        x = tf.exp(x) * (tf.cast(1.0, self.dtype) - tf.math.abs(state_vec))
        # Normalize.
        x = x / tf.reduce_sum(x, axis=-1)
        return x

    def __call__(self, game):
        """
        Sample an action from the policy and make a move.

        :param game: an instance of TicTacToe.
        """
        try:
            pdf = self.model(game.state).numpy().ravel()
            row, col = game.roll(rv_discrete(values=(np.arange(9), pdf)).rvs())
        except:
            print(self.input_W)
            print(pdf.tolist())
            exit()
        game.move(row, col)

    def update_reinforce(self, games, baseline, players, alpha, behaviors=None):
        """
        Update the policy function using the REINFORCE algorithm with a baseline and an optional behavior policy.

        :param games: a TicTacToe instance or a list of TicTacToe instances for batch learning.
        :param baseline: baseline function that takes the current state as an argument.
        :param players: either an integer from {1, -1} corresponding to 'X' or 'O', respectively, or a list of integers
            when games is a list. The number of elements of players should be equal to games and correspond to which
            player this agent is in the given game.
        :param alpha: learning rate.
        :param behaviors: a behavior policy with a model() method like the one defined in this class or a list of
            behavior policies corresponding to the behavior policy used in each game. Default is None indicating that
            the target policy is the behavior policy.
        """
        # Input preprocessing.
        if not isinstance(games, list):
            games = [games]
        if not isinstance(players, list):
            players = [players]
        if behaviors is None:
            behaviors = [None] * len(games)
        elif not isinstance(behaviors, list):
            behaviors = [behaviors] * len(games)

        # Initialize updates for each set of weights and biases.
        updates = [0.0 for _ in range(len(self.variables))]
        for game, player, behavior in zip(games, players, behaviors):
            if player == 1:
                # Agent is player 'X'.
                states = game.state_history[:len(game.action_history)][::2]
                actions = game.action_history[::2]
            elif player == -1:
                # Agent is player 'O'.
                states = game.state_history[:len(game.action_history)][1::2]
                actions = game.action_history[1::2]
            else:
                raise Exception('Invalid player choice.')

            termcode = game.check_termination()
            if termcode == player:
                # Agent won.
                r = 1
            elif termcode == -1 * player:
                # Agent lost.
                r = -1
            elif termcode == 0:
                # Draw game.
                r = 0
            else:
                raise Exception('Cannot update value function because episode has not terminated!')

            # Compute returns at each state.
            if behavior:
                # Importance sampling to reweight terminal reward.
                row, col = actions[-1]
                action_index = game.unroll(row, col)
                behave_prob = behavior.model(states[-1])[0, action_index]
                target_prob = self.model(states[-1])[0, action_index]
                ratio = target_prob / (behave_prob + np.finfo(self.np_dtype).eps)
                G = [r * ratio]
            else:
                # Behavior and target policy are the same.
                G = [r]
            for t in reversed(range(len(states) - 1)):
                if behavior:
                    # Importance sampling to reweight the returns.
                    row, col = actions[t]
                    action_index = game.unroll(row, col)
                    behave_prob = behavior.model(states[t])[0, action_index]
                    target_prob = self.model(states[t])[0, action_index]
                    ratio = target_prob / (behave_prob + np.finfo(self.np_dtype).eps)
                    G = [self.discount * G[0] * ratio] + G
                else:
                    # Behavior and target policy are the same.
                    G = [self.discount * G[0]] + G

            # Compute policy weight updates.
            for t in range(len(states)):
                delta = G[t] - baseline(states[t])
                row, col = actions[t]
                action_index = game.unroll(row, col)
                with tf.GradientTape() as tape:
                    prob = self.model(states[t])[0, action_index]
                    log_prob = tf.math.log(prob)
                grads = tape.gradient(log_prob, self.variables)

                for i in range(len(updates)):
                    updates[i] = updates[i] + delta * self.discount ** t * grads[i] / len(games)
        
        # Finally update the weights and biases of the policy.
        variables = self.variables
        for i in range(len(variables)):
            variables[i].assign(variables[i] + alpha * updates[i])

    def update_actor_critic(self, games, value_func, players, alpha, n=1, behaviors=None):
        """
        Update the policy function using the "online" actor-critic algorithm. This is not really done online since it
        would be pointless to do so in tic-tac-toe because no state can be encounted more than once in a single episode
        and therefore online learning will have the same outcome as offline learning.

        :param games: a TicTacToe instance or a list of TicTacToe instances for batch learning.
        :param value_func: function that takes the state as an argument and computes the state-value.
        :param players: either an integer from {1, -1} corresponding to 'X' or 'O', respectively, or a list of integers
            when games is a list. The number of elements of players should be equal to games and correspond to which
            player this agent is in the given game.
        :param alpha: learning rate.
        :param n: number of Monte Carlo steps to take before bootstrapping. Must be at least 1. Default is 1.
        :param behaviors: a behavior policy with a model() method like the one defined in this class or a list of
            behavior policies corresponding to the behavior policy used in each game. Default is None indicating that
            the target policy is the behavior policy.
        """
        # Input preprocessing.
        assert n >= 1
        if not isinstance(games, list):
            games = [games]
        if not isinstance(players, list):
            players = [players]
        if behaviors is None:
            behaviors = [None] * len(games)
        elif not isinstance(behaviors, list):
            behaviors = [behaviors] * len(games)

        # Initialize updates for each set of weights and biases.
        updates = [0.0 for _ in range(len(self.variables))]
        for game, player, behavior in zip(games, players, behaviors):
            if player == 1:
                # Agent is player 'X'.
                states = game.state_history[:len(game.action_history)][::2]
                actions = game.action_history[::2]
            elif player == -1:
                # Agent is player 'O'.
                states = game.state_history[:len(game.action_history)][1::2]
                actions = game.action_history[1::2]
            else:
                raise Exception('Invalid player choice.')

            termcode = game.check_termination()
            if termcode == player:
                # Agent won.
                r = 1
            elif termcode == -1 * player:
                # Agent lost.
                r = -1
            elif termcode == 0:
                # Draw game.
                r = 0
            else:
                raise Exception('Cannot update value function because episode has not terminated!')

            # Compute returns at each state.
            if behavior:
                # Importance sampling to reweight terminal reward.
                row, col = actions[-1]
                action_index = game.unroll(row, col)
                behave_prob = behavior.model(states[-1])[0, action_index]
                target_prob = self.model(states[-1])[0, action_index]
                ratio = target_prob / (behave_prob + np.finfo(self.np_dtype).eps)
                G = [r * ratio]
            else:
                # Behavior and target policy are the same.
                G = [r]
            for t in reversed(range(len(states) - 1)):
                if behavior:
                    # Importance sampling to reweight the returns.
                    row, col = actions[t]
                    action_index = game.unroll(row, col)
                    behave_prob = behavior.model(states[t])[0, action_index]
                    target_prob = self.model(states[t])[0, action_index]
                    ratio *= target_prob / (behave_prob + np.finfo(self.np_dtype).eps)
                    if t + n >= len(states):
                        # Steps reach terminal state.
                        G = [self.discount ** (len(states) - t - 1) * r * ratio] + G
                    else:
                        # Steps do not reach terminal state, approximate with state-value function.
                        G = [self.discount ** n * value_func(states[t]) * ratio] + G
                else:
                    # Behavior and target policy are the same.
                    if t + n >= len(states):
                        # Steps reach terminal state.
                        G = [self.discount ** (len(states) - t - 1) * r] + G
                    else:
                        # Steps do not reach terminal state, approximate with state-value function.
                        G = [self.discount ** n * value_func(states[t])] + G

            # Compute policy weight updates.
            for t in range(len(states)):
                delta = G[t] - value_func(states[t])
                row, col = actions[t]
                action_index = game.unroll(row, col)
                with tf.GradientTape() as tape:
                    prob = self.model(states[t])[0, action_index]
                    log_prob = tf.math.log(prob)
                grads = tape.gradient(log_prob, self.variables)

                for i in range(len(updates)):
                    updates[i] = updates[i] + delta * self.discount ** t * grads[i] / len(games)
        
        # Finally update the weights and biases of the policy.
        variables = self.variables
        for i in range(len(variables)):
            variables[i].assign(variables[i] + alpha * updates[i])


class SymBehavePolicy:

    def __init__(self, policy, mirror, nrot):
        """
        Instantiate a symmetry explointing behavior policy that will apply the agents policy at a given rotation and
        mirror transformation.

        :param policy: the agent's policy object.
        :param mirror: when True will mirror the policy's actions across the vertical axis of the game board.
        :param nrot: number of clockwise rotations to shift the policy.
        """
        self.policy = policy
        self.mirror = mirror
        self.nrot = nrot

    def model(self, state):
        """
        Compute the probability distribution over actions following transformations.

        :param state: a game state with shape (3, 3) where 1 corresponds to 'X', -1 corresponds to 'O', and 0 to an 
            empty place on the game board.
        :return: numpy array with shape (1, 9).
        """
        new_state = np.copy(state)
        new_state = np.rot90(new_state, -self.nrot)
        if self.mirror:
            new_state = new_state[:, ::-1]
        pdf = self.policy.model(new_state)
        pdf = np.reshape(pdf, (3, 3))
        if self.mirror:
            pdf = pdf[:, ::-1]
        pdf = np.rot90(pdf, self.nrot).reshape((1, -1))
        return pdf


class ValueLess:

    def __call__(self, state):
        """
        An empty value function to use as a baseline for the REINFORCE algorithm without baseline.

        :param state: unused.
        :return: 0.0.
        """
        return 0.0

    def update_monte_carlo(self, games, players):
        pass
    

class ExactValue:

    def __init__(self, discount=1.0, default=None):
        """
        Estimate a tabular value function. Exploits symmetries of the game board.

        :param discount: discount factor. Default is 1.0.
        :param default: for unexplored states, calculate a default value. By default, this is lambda state: 0.0.
        """
        self.v = {}
        self.vcounts = {}
        self.discount = discount
        self.default = default if default is not None else lambda state: 0.0

    @staticmethod
    def _h(state):
        """
        Create a unique, unmutable identifier for a given state.

        :param state: a numpy array with 9 integer elements.
        :return: a tuple of unraveled elements.
        """
        return tuple(state.ravel().tolist())

    def get_key(self, state):
        """
        Get or create an identifier for a given state.

        :param state: a numpy array with 9 integer elements.
        :return: a tuple of 9 elements corresponding to this state or an equivalent state by symmetry.
        """
        for i in range(1, 4):
            key = self._h(np.rot90(state, i))
            if key in self.v:
                return key
        tmp = state[:, ::-1]
        for i in range(4):
            key = self._h(np.rot90(tmp, i))
            if key in self.v:
                return key
        return self._h(state)

    def __call__(self, state):
        """
        Compute that value of the given state.

        :param state: a numpy array with 9 integer elements.
        :return: float value of the state.
        """
        key = self.get_key(state)
        return self.v.get(key, self.default(state))

    def update_monte_carlo(self, games, players):
        """
        Update the state-value function using the Monte Carlo algorithm.

        :param games: a TicTacToe instance or a list of TicTacToe instances for batch learning.
        :param players: either an integer from {1, -1} corresponding to 'X' or 'O', respectively, or a list of integers
            when games is a list. The number of elements of players should be equal to games and correspond to which
            player this agent is in the given game.
        """
        # Input preprocessing.
        if not isinstance(games, list):
            games = [games]
        if not isinstance(players, list):
            players = [players]

        # Iterate over games and update the state-value function.
        for game, player in zip(games, players):
            if player == 1:
                # Agent is player 'X'.
                states = game.state_history[:len(game.action_history)][::2][::-1]
            elif player == -1:
                # Agent is player 'O'.
                states = game.state_history[:len(game.action_history)][1::2][::-1]
            else:
                raise Exception('Invalid player choice.')

            termcode = game.check_termination()
            if termcode == player:
                # Agent won.
                r = 1
            elif termcode == -1 * player:
                # Agent lost.
                r = -1
            elif termcode == 0:
                # Draw game.
                r = 0
            else:
                raise Exception('Cannot update value function because episode has not terminated!')

            # Compute returns and update state-value.
            G = 0
            for state in states:
                G = self.discount * G + r
                key = self.get_key(state)
                if key not in self.v:
                    self.v[key] = 0.0
                    self.vcounts[key] = 0
                self.v[key] = (self.v[key] * self.vcounts[key] + G) / (self.vcounts[key] + 1)
                self.vcounts[key] += 1
                # All earlier states in the episode do not have a reward.
                r = 0


class ApproxValue:

    def __init__(self, discount=1.0, dtype=tf.float64):
        """
        Approximate the state-value function with a second-order polynomial. Since the function is linear, semi-
        gradient methods should be guaranteed to converge.

        :param discount: discount factor. Default is 1.0.
        :param dtype: tensorflow data type to use in calculations. Default is tf.float64.
        """
        self.discount = discount
        self.dtype = dtype
        self.np_dtype = np.float32 if dtype == tf.float32 else np.float64
        self.w = tf.Variable(np.random.randn(91, 1).astype(self.np_dtype), name='coefs')

    @tf.function
    def model(self, state):
        """
        Compute the state-value function for a given state.

        :param state: a numpy array with 9 integer elements.
        :return: float value of the state.
        """
        state_vec = tf.reshape(tf.cast(state, self.dtype), (-1, 9))
        return self.w[0] + tf.matmul(state_vec, self.w[1:10]) + \
            tf.matmul(tf.matmul(state_vec, tf.reshape(self.w[10:], [9, 9])), tf.transpose(state_vec, [1, 0]))

    def __call__(self, state):
        """
        Wraps around the model() method.

        :param state: a numpy array with 9 integer elements.
        :return: float value of the state.
        """
        return self.model(state)

    def update_monte_carlo(self, games, players, beta, target, behaviors=None):
        """
        Update the approximate state-value function with a Monte Carlo algorithm.

        :param games: a TicTacToe instance or a list of TicTacToe instances for batch learning.
        :param players: either an integer from {1, -1} corresponding to 'X' or 'O', respectively, or a list of integers
            when games is a list. The number of elements of players should be equal to games and correspond to which
            player this agent is in the given game.
        :param beta: learning rate.
        :param target: the target policy with a model() method.
        :param behaviors: a behavior policy with a model() method like the one defined in this class or a list of
            behavior policies corresponding to the behavior policy used in each game. Default is None indicating that
            the target policy is the behavior policy.
        """
        # Input preprocessing.
        if not isinstance(games, list):
            games = [games]
        if not isinstance(players, list):
            players = [players]
        if behaviors is None:
            behaviors = [None] * len(games)
        elif not isinstance(behaviors, list):
            behaviors = [behaviors] * len(games)

        # Initialize coefficient updates.
        updates = 0.0
        for game, player, behavior in zip(games, players, behaviors):
            if player == 1:
                # Agent is player 'X'.
                states = game.state_history[:len(game.action_history)][::2]
                actions = game.action_history[::2]
            elif player == -1:
                # Agent is player 'O'.
                states = game.state_history[:len(game.action_history)][1::2]
                actions = game.action_history[1::2]
            else:
                raise Exception('Invalid player choice.')

            termcode = game.check_termination()
            if termcode == player:
                # Agent won.
                r = 1
            elif termcode == -1 * player:
                # Agent lost.
                r = -1
            elif termcode == 0:
                # Draw game.
                r = 0
            else:
                raise Exception('Cannot update value function because episode has not terminated!')

            # Compute returns at each state.
            if behavior:
                # Importance sampling to reweight terminal reward.
                row, col = actions[-1]
                action_index = game.unroll(row, col)
                behave_prob = behavior.model(states[-1])[0, action_index]
                target_prob = target.model(states[-1])[0, action_index]
                ratio = target_prob / (behave_prob + np.finfo(self.np_dtype).eps)
                G = [r * ratio]
            else:
                # Behavior and target policy are the same.
                G = [r]
            for t in reversed(range(len(states) - 1)):
                if behavior:
                    # Importance sampling to reweight the returns.
                    row, col = actions[t]
                    action_index = game.unroll(row, col)
                    behave_prob = behavior.model(states[t])[0, action_index]
                    target_prob = target.model(states[t])[0, action_index]
                    ratio = target_prob / (behave_prob + np.finfo(self.np_dtype).eps)
                    G = [self.discount * G[0] * ratio] + G
                else:
                    # Behavior and target policy are the same.
                    G = [self.discount * G[0]] + G

            # Compute state-value coefficient updates.
            for t in range(len(states)):
                delta = G[t] - self.model(states[t])
                row, col = actions[t]
                action_index = game.unroll(row, col)
                with tf.GradientTape() as tape:
                    value = self.model(states[t])
                grads = tape.gradient(value, self.w)

                updates = updates + delta * grads / len(games)
        
        # Finally update the coefficients.
        self.w.assign(self.w + beta * updates)
