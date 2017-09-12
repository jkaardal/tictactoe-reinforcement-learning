import theano
import theano.tensor as T
import numpy as np
import math
import time
import matplotlib.pyplot as plt

''' tictactoe.py is a demonstration of policy gradient reinforcement learning
    that trains an artificial neural network to play the game tic-tac-toe.

'''

class ttt:
    '''The class ttt contains the necessary variables and functions to play
       tic-tac-toe. Note that ttt is only the game code and does not contain
       any training code.

    '''

    def __init__(self):
        # initialize the game
        self.state = np.zeros((3, 3), dtype=int)
        self.state_hist = np.copy(np.reshape(self.state, (3, 3, 1)))
        self.player = 1
        self.complete = False
        self.victor = 0
        
    def reinit(self):
        # restart the game
        self.state = np.zeros((3, 3), dtype=int)
        self.state_hist = np.copy(np.reshape(self.state, (3, 3, 1)))
        self.player = 1
        self.complete = False
        self.victor = 0

    def action(self, cx, cy):
        # Choose a cx horizontal position and cy vertical position on the
        # board to place an X or O. Diagram:
        #
        #      1,3|2,3|3,3
        #      -----------
        #   cy 1,2|2,2|3,2
        #      -----------
        #      1,1|2,1|3,1
        #           cx
        # 
        if not self.complete:
            if cx == 1:
                j = 0
            elif cx == 2:
                j = 1
            elif cx == 3:
                j = 2

            if cy == 1:
                i = 2
            elif cy == 2:
                i = 1
            elif cy == 3:
                i = 0

            if self.state[i, j] == 0:
                self.state[i, j] = self.player
                self.state_hist = np.copy(np.concatenate([self.state_hist, np.reshape(self.state, (3, 3, 1))], axis=2))
                self.player *= -1
                self.check_state()
                return 1
            else:
                return -1
        else:
            return 0

    def action_lin(self, c):
        # Choose a numpad position on the board and place an X or O. Diagram:
        #
        #      7|8|9
        #      -----
        #      4|5|6
        #      -----
        #      1|2|3
        #
        cy = (c-1)/3 + 1
        cx = c - 3*(cy-1)
        
        return self.action(cx, cy)

    def action_std(self, cx, cy):
        # Choose a cx horizontal position and cy inverted vertical position on
        # the board to place an X or O. Diagram:
        #
        #      1,1|2,1|3,1
        #      -----------
        #   cy 1,2|2,2|3,2
        #      -----------
        #      1,3|2,3|3,3
        #           cx
        #
        if not self.complete:
            if self.state[cy-1, cx-1] == 0:
                self.state[cy-1, cx-1] = self.player
                self.state_hist = np.copy(np.concatenate([self.state_hist, np.reshape(self.state, (3, 3, 1))], axis=2))
                self.player *= -1
                self.check_state()
                return 1
            else:
                return -1
        else:
            return 0

    def action_std_lin(self, c):
        # Choose a linear array position on the board and place an X or O.
        # Diagram:
        #
        #      1|2|3
        #      -----
        #      4|5|6
        #      -----
        #      7|8|9
        #
        cy = (c-1)/3 + 1
        cx = c - 3*(cy-1)

        return self.action_std(cx, cy)
                
    def check_state(self):
        # Check the game board to see if a winner can be declared or if the
        # game is a draw.
        for i in range(3):
            # try rows and columns
            if np.all(self.state[i,:] > 0) or np.all(self.state[:,i] > 0):
                self.complete = True
                self.victor = abs(self.player)
            elif np.all(self.state[i,:] < 0) or np.all(self.state[:,i] < 0):
                self.complete = True
                self.victor = -abs(self.player)

        if not self.complete:
            # try diagonals
            ullr = np.trace(self.state)
            llur = np.trace(self.state[:,::-1])
            if ullr >= 3*abs(self.player) or llur >= 3*abs(self.player):
                self.complete = True
                self.victor = abs(self.player)
            elif ullr <= -3*abs(self.player) or llur <= -3*abs(self.player):
                self.complete = True
                self.victor = -abs(self.player)

        if not self.complete:
            # check if the board is filled (draw game)
            if np.all(self.state.ravel() != 0):
                self.complete = True
            
    def print_state(self):
        # print the game board to screen
        lst = []
        for i in range(3):
            sublst = []
            for j in range(3):
                if self.state[i,j] > 0:
                    sublst.append("X")
                elif self.state[i,j] < 0:
                    sublst.append("O")
                else:
                    sublst.append(" ")
            lst.append(sublst)
        print "|".join(lst[0])
        print "-----"
        print "|".join(lst[1])
        print "-----"
        print "|".join(lst[2])
        print ""
        

def sigmoid(W, X):
    # logistic activation function
    return 1.0/(1.0 + T.exp(-T.tile(W[:,0].reshape((W.shape[0], 1)), (1, X.shape[1])) - T.dot(W[:,1:], X)))


def relu(W, X):
    # rectified-linear unit activation function
    L = T.tile(W[:,0].reshape((W.shape[0], 1)), (1, X.shape[1])) + T.dot(W[:,1:], X)
    return T.switch(L>0.0, L, 0.0)


def init_dev():
    # initialize 'device' variables (not used)
    X = T.matrix('X')
    W_in = theano.shared(np.array([]), name='W_in')
    H = theano.shared(np.array([]), name='H')
    W_out = theano.shared(np.array([]), name='W_out')

    return (X, W_in, H, W_out)
    

def build_ffnn(X, W_in, H, W_out, **kwargs):
    # Build a feed-forward neural network with input X,
    # first hidden layer W_in, intermediate hidden layers
    # H, and output hidden layer W_out.
    activation = kwargs.get("activation", relu)
    
    def feed_forward(idx, z, H):
        z = activation(H[:,:,idx], z)
        return z
        
    z_in = activation(W_in, X)

    if H is not None:
        z_mid, updates = theano.scan(
            fn=feed_forward,
            outputs_info=z_in,
            sequences=T.arange(H.shape[2]),
            non_sequences=H,
        )
    else:
        z_mid = [z_in]

    z_out = activation(W_out, z_mid[-1])
    
    return z_out


def weights_to_vec(*args):
    # not used
    x = args[0].ravel()
    for i in range(1, len(args)):
        x = T.concatenate([x, args[i].ravel()], axis=0)

    return x


def discounted_rewards(game, player, gamma, success):
    # generate reward vector for the game
    r = np.zeros((game.state_hist.shape[2], 1))
    if player > 0:
        r = r[:-1:2]
    else:
        r = r[1:-1:2]
    if success == -1:
        # infeasible move, penalize (not used)
        r[-1] = -1
        return r
    if player == game.victor:
        # if player wins, give reward of 1
        reward_val = 1
    elif -player == game.victor:
        # if player loses, give penalty of -1
        reward_val = -1
    else:
        # if the game is a draw, return 0 rewards/penalties
        return r #np.ones(r.shape)*0.1
    # Since this is a delayed reward problem, set the reward of
    # the terminal state to reward_val.
    r[-1] = reward_val
    for i in reversed(range(r.size-1)):
        # use discounting to estimate the correlation length of
        # the reward to prior actions
        r[i] += r[i+1]*gamma
    # emphasize quick wins/slow losses by dividing by the game length.
    r /= r.shape[1]
    
    return r


def generate_data(game, player, gamma, success):
    # Generate training data from the board state history of a game.
    X = np.reshape(game.state_hist, (9, game.state_hist.shape[2]))
    # only take as input the state prior to an action being taken by the player
    if player > 0:
        X_in = X[:,:-1:2]
        X_out = X[:,1::2]
    else:
        X_in = X[:,1:-1:2]
        X_out = X[:,2::2]
    # exploit rotational invariance to generate extra data
    for i in range(1,4):
        X = np.reshape(np.rot90(game.state_hist, k=i), (9, game.state_hist.shape[2]))
        if player > 0:
            X_in = np.concatenate([X_in, X[:,:-1:2]], axis=1)
            X_out = np.concatenate([X_out, X[:,1::2]], axis=1)
        else:
            X_in = np.concatenate([X_in, X[:,1:-1:2]], axis=1)
            X_out = np.concatenate([X_out, X[:,2::2]], axis=1)
    # exploit symmetry invariance to generate extra data
    for i in range(0,4):
        X = np.reshape(np.rot90(game.state_hist[:,::-1], k=i), (9, game.state_hist.shape[2]))
        if player > 0:
            X_in = np.concatenate([X_in, X[:,:-1:2]], axis=1)
            X_out = np.concatenate([X_out, X[:,1::2]], axis=1)
        else:
            X_in = np.concatenate([X_in, X[:,1:-1:2]], axis=1)
            X_out = np.concatenate([X_out, X[:,2::2]], axis=1)

    # generate the decision by taking the absolute difference between the prior
    # and posterior states.
    X_out = np.abs(X_out-X_in)

    # Multiply the input data by the player label such that player always corresponds
    # to 1 while the adversary corresponds to -1 (this is not strictly necessary, but
    # might be helpful to simplify the network; beware when removing this line that
    # this modification occurs elsewhere in the code as well).
    X_in *= player

    # generate rewards vector
    rewards = discounted_rewards(game, player, gamma, success)
    # extend the rewards to the invariant training data
    rewards = np.tile(rewards.reshape((1, rewards.size)), (1, X_out.shape[1]/rewards.size)).ravel()
    
    return (X_in, X_out, rewards)


def update_net(X_in, X_out, rewards, weight_update, RMS_update=None):
    # Update the network weights and the RMSprop learning rate adjustments (if
    # applicable).
    if RMS_update is not None:
        RMS_update(X_in, X_out, rewards)

    weight_update(X_in, X_out, rewards)
    

def random_actor(game, **kwargs):
    # Player chooses a random feasible action.
    z = np.random.rand(9, 1)
    z[game.state.ravel() != 0] = 0
    zc = np.cumsum(z)/np.sum(z)
    c = int(np.sum(np.random.rand() >= np.concatenate([np.zeros((1,)), zc[:-1]], axis=0)))
    success = game.action_std_lin(c)

    return success


def trained_actor(game, **kwargs):
    # Player uses the policy network to generate a probability distribution
    # of action choices.
    print_prob = kwargs.get("print_prob", False)
    policy = kwargs.get("policy", None)
    epsilon = kwargs.get("epsilon", 0.0)
    if np.random.rand() > (1.0-epsilon) or policy is None:
        # if 'epsilon-greedy' steps are taken, choose a random feasible action.
        success = random_actor(game)
    else:
        # get the current game board state
        x = game.state.ravel().reshape((9,1))
        # change the sign of the state such that player is always 1 and opponent
        # is always -1 and generate the probability distribution of action choices.
        z = policy(game.player*x)
        if print_prob:
            # print the action probability distribution
            print z
        # set the probability of infeasible actions to zero
        z[game.state.ravel() != 0] = 0
        # generate the normalized cumulative distribution function from the action
        # probability distribution
        if np.sum(z) == 0.0 and np.sum(game.state.ravel() == 0) > 0:
            z[game.state.ravel() == 0] = np.random.rand(np.sum(game.state.ravel() == 0),1)
        zc = np.cumsum(z)/np.sum(z)
        # draw an action from the action probability distribution
        c = int(np.sum(np.random.rand() >= np.concatenate([np.zeros((1,)), zc[:-1]], axis=0)))
        if np.any(np.isnan(z)):
            raise Exception('NaN output probability detected. Terminating.')
        success = game.action_std_lin(c)

    return success


def player_actor(game, **kwargs):
    # Human player can play by using the numpad.
    x = game.state[::-1,:].ravel()
    while True:
        c = input('Choose an action (1-9): ')
        if x[c-1] == 0:
            break
        else:
            print 'Invalid action! Try again.'
    success = game.action_lin(c)

    return success
    

def init_batches(float_dtype):
    # initialize batch training set arrays
    batch_X_in = np.array([], dtype=float_dtype).reshape((9,0))
    batch_X_out = np.array([], dtype=float_dtype).reshape((9,0))
    batch_rewards = np.array([], dtype=float_dtype).reshape((1,0))

    return (batch_X_in, batch_X_out, batch_rewards)


def update_batches(batch_X_in, batch_X_out, batch_rewards, X_in, X_out, rewards):
    # update batch training set arrays
    batch_X_in = np.concatenate([batch_X_in, X_in], axis=1)
    batch_X_out = np.concatenate([batch_X_out, X_out], axis=1)
    batch_rewards = np.concatenate([batch_rewards, rewards.reshape((1, rewards.size))], axis=1)

    return (batch_X_in, batch_X_out, batch_rewards)


def main():

    # total number of games to play
    num_games = 2E6
    
    # data type/machine precision
    float_dtype = np.float64
    eps = np.finfo(float_dtype).eps
    
    # set learning rate
    alpha = theano.shared(float_dtype(0.001), name='alpha')
    #schedule_alpha = lambda x, ngames, periods: 0.01*np.cos(np.pi/num_games*((ngames*periods) % (2*num_games))) ** 2
    schedule_alpha = lambda x, ngames, periods: 0.001/(1+10.0*(ngames-1)/num_games)
    periods = 1

    # set batch size
    batchsize = 100

    # use a residual bypass (simple residual neural network to improve convergence)
    bypass = True
    
    # use 'epsilon-greedy' exploration (not necessary to use)
    epsilon = 0.5                      # exploration probability (epsilon=0.0 disables random exploration)
    schedule_epsilon = lambda x, rate: rate*x  # annealing schedule for the probability of random actions
    rate = 0.99999769741               # rate multiplier for the annealing schedule

    # set actors
    player1 = random_actor   # neural network does NOT train on player1 actions
    player2 = trained_actor  # neural network trains on player2 actions
    
    # discount factor
    gamma = 0.95

    # file I/O
    resume = True                        # if True, weights are initialized from file
    save_results = True                  # if True, weights and stats are saved to file
    weight_file_input = "./weights.npz"  # weights input file
    weight_file_output = "./weights.npz" # weights output file
    stat_file = "./stats.npz"            # wins/losses/draws statistics file
    plot_file = "./wins.png"
    
    # intialize weights
    if resume:
        # load weights from input file
        weights = np.load(weight_file_input)
        W_in = weights['W_in']
        H = weights['H']
        W_out = weights['W_out']
        W_final = weights['W_final']
        hid_units = H.shape[1]-1
        hid_layers = H.shape[2]
        if W_final.shape[1]-1 > hid_units:
            bypass = True
    else:
        # initilize weights to normally distributed random numbers
        hid_units = 200
        hid_layers = 1
        N = 10*hid_units + (hid_units+1)*hid_units*hid_layers + (hid_units+1)*hid_units + 9*hid_units + 9*10
        W_in = np.random.randn(hid_units, 10)*np.sqrt(1.0/10)
        H = np.random.randn(hid_units, hid_units+1, hid_layers)*np.sqrt(1.0/(hid_units+1))
        W_out = np.random.randn(hid_units, hid_units+1)*np.sqrt(1.0/(hid_units+1))
        if bypass:
            # include residual bypass
            W_final = np.random.randn(9, hid_units+10)*np.sqrt(1.0/(hid_units+1))
        else:
            # no residual bypass
            W_final = np.random.randn(9, hid_units+1)*np.sqrt(1.0/(hid_units+1))
    
    # declare theano 'device' variables
    X_in = T.matrix('X_in')                                              # training data argument
    X_out = T.matrix('X_out')                                            # action data argument
    W_in = theano.shared(W_in.astype(float_dtype), name='W_in')          # first hidden layer
    H = theano.shared(H.astype(float_dtype), name='H')                   # intermediate hidden layers
    W_out = theano.shared(W_out.astype(float_dtype), name='W_out')       # last intermediate hidden layer
    W_final = theano.shared(W_final.astype(float_dtype), name='W_final') # final hidden layer (with residual input, if applicable)
    reward_weight = T.matrix('reward_weight')                            # reward weighting argument

    # Build the feed-forward neural network with rectified-linear units activation function. Note that
    # relu is used instead of sigmoid activation functions because they are less susceptible to the
    # vanishing/exploding gradient problems.
    Z = build_ffnn(X_in, W_in, H, W_out, activation=relu)

    # build final hidden layer
    if bypass:
        # with residual bypass
        X_final = T.concatenate([Z, X_in], axis=0) # note that the input is passed directly into this final layer
    else:
        # without residual bypass
        X_final = Z

    # output layer activations
    Z_final = sigmoid(W_final, X_final) # uses a sigmoid in this case such that p(action) \in (0.0, 1.0)
        
    # likelihood ratio objective function
    f = -T.sum( (((X_out*T.log(Z_final+eps))) * T.tile(reward_weight.reshape((1, reward_weight.size)), (9, 1))).ravel() ) / batchsize

    # generate theano expressions for the gradients of the weights
    gW_in, gH, gW_out, gW_final = T.grad(f, [W_in, H, W_out, W_final])

    # RMSprop (tailors the learning rate to each weight)
    RMS_decay = theano.shared(float_dtype(0.9), name='RMS_decay') # decay rate of historic gradient weightings
    vW_in = theano.shared(np.ones((hid_units, 10), dtype=float_dtype), name='vW_in')
    vH = theano.shared(np.ones((hid_units, hid_units+1, hid_layers), dtype=float_dtype), name='vH')
    vW_out = theano.shared(np.ones((hid_units, hid_units+1), dtype=float_dtype), name='vW_out')
    if bypass:
        # with residual bypass
        vW_final = theano.shared(np.ones((9, hid_units+10), dtype=float_dtype), name='vW_final')
    else:
        # without residual bypass
        vW_final = theano.shared(np.ones((9, hid_units+1), dtype=float_dtype), name='vW_final')    

    # reweight the gradients using RMSprop
    W_in_update = gW_in/T.sqrt(vW_in + 1.0E-6)
    H_update = gH/T.sqrt(vH + 1.0E-6)
    W_out_update = gW_out/T.sqrt(vW_out + 1.0E-6)
    W_final_update = gW_final/T.sqrt(vW_final + 1.0E-6)

    # update the RMSprop weightings
    RMS_update = theano.function(inputs=[X_in, X_out, reward_weight], updates=((vW_in, RMS_decay*vW_in + (1.0-RMS_decay)*gW_in**2), (vH, RMS_decay*vH + (1.0-RMS_decay)*gH**2), (vW_out, RMS_decay*vW_out + (1.0-RMS_decay)*gW_out**2), (vW_final, RMS_decay*vW_final + (1.0-RMS_decay)*gW_final**2)))

    # compile th policy and weight update functions
    policy = theano.function(inputs=[X_in], outputs=Z_final, on_unused_input='ignore')
    weight_update = theano.function(inputs=[X_in, X_out, reward_weight], outputs=f, updates=((W_in, W_in - alpha*W_in_update), (H, H - alpha*H_update), (W_out, W_out - alpha*W_out_update), (W_final, W_final - alpha*W_final_update)))

    # reinforcement learning algorithm
    nwins = 0
    ndraws = 0
    track_length = 500
    track_wins = np.ones((track_length,))*np.nan                                # track a running average of past wins
    track_wins_draws = np.ones((track_length,))*np.nan                          # track a running average of past wins and draws
    game = ttt()                                                       # initialize the tic-tac-toe game class
    plt.ion()
    plot_interval = 500
    batch_X_in, batch_X_out, batch_rewards = init_batches(float_dtype) # initialize batch arrays
    for ngames in range(1, int(num_games+1)):
        # play the game until ngames > num_games
        print "GAME " + str(ngames)

        # choose which player is X and O and play game
        rval = np.random.rand()
        if rval > 0.5:
            p2 = -1
            while not game.complete:
                # player 1 move
                c1 = player1(game, policy=policy, epsilon=epsilon, print_prob=False)
                # player 2 move
                if not game.complete:
                    game.print_state()
                    c2 = player2(game, policy=policy, epsilon=epsilon, print_prob=True)
                if c2 == -1:
                    break
        else:
            p2 = 1
            while not game.complete:
                # player 2 move
                game.print_state()
                c2 = player2(game, policy=policy, epsilon=epsilon, print_prob=True)
                if c2 == -1:
                    break
                # player 1 move
                if not game.complete:
                    c1 = player1(game, policy=policy, epsilon=epsilon, print_prob=False)

        # game is complete, print the board state
        game.print_state()
        if c2 == -1:
            # not used
            print "Infeasible choice made."
        # generate training data from the last game
        X_in, X_out, rewards = generate_data(game, p2, gamma, c2)
        # update the batch training data with the new training data
        batch_X_in, batch_X_out, batch_rewards = update_batches(batch_X_in, batch_X_out, batch_rewards, X_in, X_out, rewards)

        if not (ngames % batchsize):
            # once batchsize games have been played, update the network weights
            print "Updating weights..."
            # center and normalize the rewards
            batch_rewards -= np.mean(batch_rewards)
            batch_rewards /= np.std(batch_rewards)
            # update network weights
            update_net(batch_X_in, batch_X_out, batch_rewards, weight_update, RMS_update)
            # empty the batch arrays
            batch_X_in, batch_X_out, batch_rewards = init_batches(float_dtype)

        # track wins/losses/draws for player2
        if game.victor == p2:
            nwins += 1
            if ngames <= track_wins.size:
                track_wins[ngames-1] = 1.0
                track_wins_draws[ngames-1] = 1.0
            else:
                track_wins[:-1] = track_wins[1:]
                track_wins[-1] = 1.0
                track_wins_draws[:-1] = track_wins_draws[1:]
                track_wins_draws[-1] = 1.0
        elif game.victor == 0:
            ndraws += 1
            if ngames <= track_wins.size:
                track_wins_draws[ngames-1] = 1.0
            else:
                track_wins_draws[:-1] = track_wins_draws[1:]
                track_wins_draws[-1] = 1.0
        else:
            if ngames <= track_wins.size:
                track_wins[ngames-1] = 0.0
                track_wins_draws[ngames-1] = 0.0
            else:
                track_wins[:-1] = track_wins[1:]
                track_wins[-1] = 0.0
                track_wins_draws[:-1] = track_wins_draws[1:]
                track_wins_draws[-1] = 0.0

        print "Learning rate: " + str(alpha.get_value()) + ", exploration probability: " + str(epsilon)
        print "Computer win fraction: " + str(1.0*nwins/ngames) + ", Computer wins + draws fraction: " + str(1.0*(nwins+ndraws)/ngames)
        print "Recent computer win fraction: " + str(np.nanmean(track_wins)) + ", Recent computer wins + draws fraction: " + str(np.nanmean(track_wins_draws))

        # reset the game
        game.reinit()

        # update the 'epsilon-greedy' exploration probability
        epsilon = schedule_epsilon(epsilon, rate)

        # plot recent win ratio
        if ngames == plot_interval:
            h1, = plt.plot(ngames, np.nanmean(track_wins), 'r')
            plt.draw()
        else:
            if not (ngames % plot_interval):
               h1.set_xdata(np.append(h1.get_xdata(), ngames))
               h1.set_ydata(np.append(h1.get_ydata(), np.nanmean(track_wins)))
               ax = plt.gca()
               ax.relim()
               ax.autoscale_view()
               plt.draw()
               plt.pause(0.1)

        alpha.set_value(schedule_alpha(alpha.get_value(), ngames, periods))

    if save_results:
        # save weights and stats as .npz files
        np.savez(weight_file_output, W_in=W_in.get_value(), H=H.get_value(), W_out=W_out.get_value(), W_final=W_final.get_value())
        np.savez(stat_file, nwins=nwins, ndraws=ndraws, ngames=ngames)
        
        plt.savefig(plot_file)

if __name__ == "__main__":
    main()
