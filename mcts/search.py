import copy
import torch

from mcts.node import MCTSNode


class MCTS:
    """
    Monte Carlo Tree Search using Neural Network guidance
    """

    def __init__(self, net, encoder, simulations=50, c_puct=1.4):

        self.net = net
        self.encoder = encoder

        self.simulations = simulations
        self.c_puct = c_puct


    def run(self, env):

        """
        Run MCTS from current environment state
        """

        root = MCTSNode(parent=None, prior=1.0)

        # Expand root first
        self._expand_root(root, env)


        # Run simulations
        for _ in range(self.simulations):

            env_copy = copy.deepcopy(env)

            node = root

            # -----------------------
            # SELECTION
            # -----------------------

            while not node.is_leaf():

                action, node = node.best_child(self.c_puct)

                move = self.encoder.decode(action)

                env_copy.step(move)


            # -----------------------
            # EVALUATION
            # -----------------------

            value = self._evaluate(env_copy)


            # -----------------------
            # EXPANSION
            # -----------------------

            if not env_copy.board.is_game_over():

                priors = self._policy(env_copy)

                node.expand(priors)


            # -----------------------
            # BACKUP
            # -----------------------

            node.backup(value)


        return root


    def _expand_root(self, root, env):

        priors = self._policy(env)

        root.expand(priors)


    def _policy(self, env):

        """
        Get NN policy for state
        Returns dict: action -> prob
        """

        state = env.get_state()

        s = torch.tensor(state).float().unsqueeze(0)

        with torch.no_grad():
            policy, _ = self.net(s)

        policy = policy.squeeze().cpu()


        legal_moves = env.legal_moves()

        priors = {}


        for m in legal_moves:

            idx = self.encoder.encode(m)

            if idx is not None:
                priors[idx] = policy[idx].item()


        # Normalize
        total = sum(priors.values())

        if total > 0:
            for k in priors:
                priors[k] /= total
        else:
            # fallback uniform
            n = len(priors)

            for k in priors:
                priors[k] = 1.0 / n


        return priors


    def _evaluate(self, env):

        """
        Value evaluation
        """

        state = env.get_state()

        s = torch.tensor(state).float().unsqueeze(0)

        with torch.no_grad():
            _, value = self.net(s)

        return value.item()
