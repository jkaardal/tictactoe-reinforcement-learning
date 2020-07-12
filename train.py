import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from .game import TicTacToe
from .agent import ApproxPolicy, ValueLess, ExactValue, RandomPolicy, SymBehavePolicy, ApproxValue
from .util import get_all_analogues


if __name__ == '__main__':
    policy_choices = [
        'reinforce',
        'reinforce-with-baseline',
        'actor-critic'
    ]
    policy_choices = policy_choices + [c.upper() for c in policy_choices]

    value_choices = [
        'tabular-monte-carlo',
        'approx-monte-carlo',
    ]
    value_choices = value_choices + [c.upper() for c in value_choices]

    parser = argparse.ArgumentParser(description='Train a Tic-Tac-Toe agent using reinforcement learning.')
    parser.add_argument('--train-samples', action='store', default=100, type=int, 
                        help='Number of games per epoch used for training.', dest='train_samples')
    parser.add_argument('--eval-samples', action='store', default=300, type=int,
                        help='Number of games per epoch used for evaluation.', dest='eval_samples')
    parser.add_argument('--num-epochs', action='store', default=3000, type=int,
                        help='Total number of epochs used to train agent.', dest='num_epochs')
    parser.add_argument('--discount', action='store', default=1.0, type=float, 
                        help='Discount factor.', dest='discount')
    parser.add_argument('--policy-hidden-layers', action='store', default=0, type=int, dest='nlayers',
                        help='Number of hidden layers in the approximate policy.')
    parser.add_argument('--policy-hidden-units', action='store', default=256, type=int, dest='nunits',
                        help='Number of units in each hidden layer.')
    parser.add_argument('--policy-skips', action='store', default=0, type=int, dest='nskips',
                        help='Number of layers to skip between in residual connection.')
    parser.add_argument('--policy-learning-rate', action='store', default=1.0e-4, type=float,
                        help='Policy gradients learning rate.', dest='policy_alpha')
    parser.add_argument('--policy-update-algo', action='store', default='reinforce', type=str,
                        choices=policy_choices, help='Policy update algorithm.', dest='policy_update_algo')
    parser.add_argument('--policy-n-steps', action='store', default=1, type=int, dest='nsteps',
                        help='Number of steps in actor-critic update.')
    parser.add_argument('--value-algo', action='store', default='tabular-monte-carlo', type=str,
                        choices=value_choices, help='State-value function algorihtm', dest='value_algo')
    parser.add_argument('--value-learning-rate', action='store', default=1.0e-2, type=float, dest='value_beta',
                        help='Learning rate for updating approximate state-value function.')
    parser.add_argument('--rand-opponent', action='store_true', help='Play a random opponent instead of self-play',
                        dest='rand_opponent')
    parser.add_argument('--enable-sym', action='store_true', dest='enable_sym',
                        help='Enable symmetry exploitation in approximate methods.')
    parser.add_argument('--plot', action='store_true', help='Plot training progress.')

    args = parser.parse_args()
    policy_update_algo = args.policy_update_algo.lower()
    value_algo = args.value_algo.lower()

    # Initialize game object.
    game = TicTacToe()

    # Initialize agent policy and state-value function (if applicable).
    policy = ApproxPolicy(args.nlayers, args.nunits, args.nskips, discount=args.discount)
    if policy_update_algo in ['reinforce-with-baseline', 'actor-critic']:
        if value_algo == 'tabular-monte-carlo':
            # Use tabular state-value function as baseline for REINFORCE algorithm.
            value = ExactValue(discount=args.discount)
        else:
            # Use approximate state-value function.
            value = ApproxValue(discount=args.discount)
    else:
        # use zero baseline for REINFORCE algorithm
        value = ValueLess()

    # Initialize behavior policies to take advantage of symmetry.
    behaviors = []
    for i in range(4):
        behaviors.append(SymBehavePolicy(policy, False, i))
    for i in range(4):
        behaviors.append(SymBehavePolicy(policy, True, i))

    # Initialize random policy for evaluating agent.
    eval_policy = RandomPolicy()
    win_rate = []

    print(f'Epoch 0')
    # Perform an initial evaluation of the policy.
    win_loss_draw = []
    for sample in range(args.eval_samples):
        first = True
        player = np.random.choice([-1, 1])
        while game.check_termination() is None:
            if player == 1 or not first:
                policy(game)

            if game.check_termination() is None:
                eval_policy(game)
            first = False

        winner = game.check_termination()
        if winner == player:
            win_loss_draw.append([1, 0, 0])
        elif winner == -1 * player:
            win_loss_draw.append([0, 1, 0])
        else:
            win_loss_draw.append([0, 0, 1])

        game.reset()

    win_loss_draw = np.array(win_loss_draw)
    print(np.sum(win_loss_draw, axis=0), np.sum(win_loss_draw[:, 0]) / np.sum(win_loss_draw[:, :2]))
    win_rate.append(np.sum(win_loss_draw[:, 0]) / np.sum(win_loss_draw[:, :2]))

    if args.plot:
        # Optionally plot progress of training.
        plt.ion()
        fig = plt.figure()
        ax = plt.gca()
        line, = ax.plot(np.arange(len(win_rate)), win_rate)
        ax.set_autoscale_on(True)
        plt.draw()
        plt.pause(0.01)

    # Train the policy.
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}')
        if not args.rand_opponent:
            # Agent plays against itself.
            for sample in range(args.train_samples):
                while game.check_termination() is None:
                    # While the game has not terminated (win, loss, or draw), keep moving.
                    policy(game)
                    if game.check_termination() is None:
                        policy(game)
                
                # Recover the winner of the game.
                winner = game.check_termination()

                if not args.enable_sym:
                    # Symmetry is not being exploited.
                    if value_algo == 'tabular-monte-carlo':
                        # Update tabular state-value function.
                        value.update_monte_carlo([game, game], [1, -1])
                    else:
                        # Update approximate state-value function.
                        value.update_monte_carlo([game, game], [1, -1], args.value_beta, policy)
                    
                    if policy_update_algo.startswith('reinforce'):
                        # Update REINFORCE policies.
                        policy.update_reinforce([game, game], value, [1, -1], args.policy_alpha)
                    else:
                        # Update actor-critic policy.
                        policy.update_actor_critic([game, game], value, [1, -1], args.policy_alpha, args.nsteps)
                else:
                    # Exploit symmetry.
                    analogues = get_all_analogues(game)

                    if value_algo == 'tabular-monte-carlo':
                        # Update tabular state-value function.
                        value.update_monte_carlo(
                            analogues + analogues,
                            [1] * len(analogues) + [-1] * len(analogues),
                        )
                    else:
                        # Update approximate state-value function.
                        value.update_monte_carlo(
                            analogues + analogues,
                            [1] * len(analogues) + [-1] * len(analogues),
                            args.value_beta,
                            policy,
                            behaviors + behaviors,
                        )

                    if policy_update_algo.startswith('reinforce'):
                        # Update REINFORCE policy.
                        policy.update_reinforce(
                            analogues + analogues,
                            value,
                            [1] * len(analogues) + [-1] * len(analogues), 
                            args.policy_alpha,
                            behaviors + behaviors
                        )
                    else:
                        # Update actor-critic policy.
                        policy.update_actor_critic(
                            analogues + analogues,
                            value,
                            [1] * len(analogues) + [-1] * len(analogues), 
                            args.policy_alpha,
                            args.nsteps,
                            behaviors + behaviors
                        )

                game.reset()
        else:
            # Play against an opponent with a random policy.
            for sample in range(args.train_samples):
                # Randomly choose which player is the agent.
                player = np.random.choice([-1, 1])
                first = True
                while game.check_termination() is None:
                    # While the game has not terminated yet, make moves.
                    if player == 1 or not first:
                        policy(game)

                    if game.check_termination() is None:
                        eval_policy(game)
                    first = False

                if not args.enable_sym:
                    # Symmetry is not being exploited.
                    if value_algo == 'tabular-monte-carlo':
                        # Update tabular state-value function.
                        value.update_monte_carlo(game, player)
                    else:
                        # Update approximate state-value function.
                        value.update_monte_carlo(game, player, args.value_beta, policy)

                    if policy_update_algo.startswith('reinforce'):
                        # Update REINFORCE policy.
                        policy.update_reinforce(game, value, player, args.policy_alpha)
                    else:
                        # Update actor-critic policy.
                        policy.update_actor_critic(game, value, player, args.policy_alpha, args.nsteps)
                else:
                    # Exploit symmetry.
                    analogues = get_all_analogues(game)

                    if value_algo == 'tabular-monte-carlo':
                        # Update tabular state-value function.
                        value.update_monte_carlo(
                            analogues,
                            [player] * len(analogues),
                        )
                    else:
                        # Update approximate state-value function.
                        value.update_monte_carlo(
                            analogues,
                            [player] * len(analogues),
                            args.value_beta,
                            policy,
                            behaviors,
                        )

                    if policy_update_algo.startswith('reinforce'):
                        # Update REINFORCE policy.
                        policy.update_reinforce(
                            analogues,
                            value,
                            [player] * len(analogues), 
                            args.policy_alpha,
                            behaviors
                        )
                    else:
                        # Update actor-critic policy.
                        policy.update_actor_critic(
                            analogues,
                            value,
                            [player] * len(analogues), 
                            args.policy_alpha,
                            args.nsteps,
                            behaviors
                        )

                game.reset()               

        # Evaluate the policy.
        win_loss_draw = []
        for sample in range(args.eval_samples):
            first = True
            player = np.random.choice([-1, 1])
            while game.check_termination() is None:
                if player == 1 or not first:
                    policy(game)

                if game.check_termination() is None:
                    eval_policy(game)
                first = False

            winner = game.check_termination()
            if winner == player:
                win_loss_draw.append([1, 0, 0])
            elif winner == -1 * player:
                win_loss_draw.append([0, 1, 0])
            else:
                win_loss_draw.append([0, 0, 1])

            game.reset()

        win_loss_draw = np.array(win_loss_draw)
        print(np.sum(win_loss_draw, axis=0), np.sum(win_loss_draw[:, 0]) / np.sum(win_loss_draw[:, :2]))
        win_rate.append(np.sum(win_loss_draw[:, 0]) / np.sum(win_loss_draw[:, :2]))
        if args.plot:
            # Update plots.
            line.set_xdata(np.arange(len(win_rate)))
            line.set_ydata(win_rate)
            ax.relim()
            ax.autoscale_view(True, True, True)
            plt.draw()
            plt.pause(0.01)

    input('Press enter to exit.')