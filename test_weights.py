import numpy as np
import theano
import theano.tensor as T
import tictactoe

# data types
float_dtype = np.float64

# load weights
weights = np.load('./weights.npz')
W_in = weights['W_in']
H = weights['H']
W_out = weights['W_out']
W_final = weights['W_final']

X_in = T.matrix('X_in')
X_out = T.matrix('X_out')
W_in = theano.shared(W_in.astype(float_dtype), name='W_in')
H = theano.shared(H.astype(float_dtype), name='H')
W_out = theano.shared(W_out.astype(float_dtype), name='W_out')
W_final = theano.shared(W_final.astype(float_dtype), name='W_final')

# build policy network
Z = tictactoe.build_ffnn(X_in, W_in, H, W_out)
X_final = T.concatenate([Z, X_in], axis=0)
Z_final = tictactoe.sigmoid(W_final, X_final)

# compile policy network
policy = theano.function(inputs=[X_in], outputs=Z_final, on_unused_input='ignore')

ngames = 1
nwins = 0
ndraws = 0
game = tictactoe.ttt()
while True:

    if np.random.rand() > 0.5:
        while not game.complete:
            # player's turn
            game.print_state()
            # human input
            x = game.state[::-1,:].ravel()
            #x[game.state[::-1,:].ravel() != 0] = np.nan
            print x
            while True:
                c = input('Choose an action (1-9): ')
                if x[c-1] == 0:
                    break
                else:
                    print "Invalid action!"
            # random feasible input
            #z = np.random.rand(9, 1)
            #z[game.state[::-1,:].ravel() != 0] = 0
            #zc = np.cumsum(z)/np.sum(z)
            #c = int(np.sum(np.random.rand() >= np.concatenate([np.zeros((1,)), zc[:-1]], axis=0)))
            game.action_lin(c)
            # computer's turn
            if not game.complete:
                x = -game.state.reshape((9, 1))
                #x = np.concatenate([x, -np.ones((1,1))], axis=0)
                z = policy(x)
                z[game.state.ravel() != 0] = 0
                print z
                c = np.argmax(z)+1
                #zc = np.cumsum(z)/np.sum(z)
                #c = int(np.sum(np.random.rand() >= np.concatenate([np.zeros((1,)), zc[:-1]], axis=0)))
                game.action_std_lin(c)

        if game.victor < 0:
            nwins += 1
        elif game.victor == 0:
            ndraws += 1
    else:
        while not game.complete:
            # computer's turn
            x = game.state.reshape((9, 1))
            #x = np.concatenate([x, np.ones((1,1))], axis=0)
            z = policy(x)
            z[game.state.ravel() != 0] = 0
            print z
            c = np.argmax(z)+1
            #zc = np.cumsum(z)/np.sum(z)
            #c = int(np.sum(np.random.rand() >= np.concatenate([np.zeros((1,)), zc[:-1]], axis=0)))
            game.action_std_lin(c)
            if not game.complete:
                # player's turn
                game.print_state()
                # human input
                x = np.copy(game.state[::-1,:].ravel())
                print x
                while True:
                    c = input('Choose an action (1-9): ')
                    if x[c-1] == 0:
                        break
                    else:
                        print "Invalid action!"
                # random feasible input
                #z = np.random.rand(9, 1)
                #z[game.state[::-1,:].ravel() != 0] = 0
                #zc = np.cumsum(z)/np.sum(z)
                #c = int(np.sum(np.random.rand() >= np.concatenate([np.zeros((1,)), zc[:-1]], axis=0)))
                game.action_lin(c)
        if game.victor > 0:
            nwins += 1
        elif game.victor == 0:
            ndraws += 1

    game.print_state()
    print "Computer win fraction: " + str(1.0*nwins/ngames) + ", Computer wins + draws fraction: " + str(1.0*(nwins+ndraws)/ngames)

    game.reinit()
    ngames += 1
