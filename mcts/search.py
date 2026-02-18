import copy
import torch

from mcts.node import MCTSNode


class MCTS:
    """
    Monte Carlo Tree Search using Neural Network guidance
    + Reasoning extraction
    """

    def __init__(self, net, encoder, simulations=50, c_puct=1.4):

        self.net = net
        self.encoder = encoder

        self.simulations = simulations
        self.c_puct = c_puct


    # ==================================================
    # MAIN SEARCH
    # ==================================================

    def run(self, env):
        """
        Run MCTS from current environment state

        Returns:
            root (MCTSNode)
            reasoning (list of dict)
        """

        root = MCTSNode(parent=None, prior=1.0)

        # Expand root
        self._expand_root(root, env)


        # -----------------------
        # SIMULATIONS
        # -----------------------

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


        # -----------------------
        # EXTRACT REASONING
        # -----------------------

        reasoning = self._extract_reasoning(root)

        return root, reasoning


    # ==================================================
    # REASONING
    # ==================================================

    def _extract_reasoning(self, root, top_k=5):
        """
        Extract top candidate moves for explanation

        Returns:
        [
          {
            "move": "e2e4",
            "visits": 42,
            "Q": 0.31,
            "prior": 0.18
          },
          ...
        ]
        """

        info = []

        for action, child in root.children.items():

            move = self.encoder.decode(action)

            info.append({
                "move": str(move),
                "visits": int(child.N),
                "Q": round(child.Q, 4),
                "prior": round(child.P, 4)
            })


        # Sort by visits (importance)
        info.sort(key=lambda x: x["visits"], reverse=True)

        return info[:top_k]


    # ==================================================
    # ROOT EXPANSION
    # ==================================================

    def _expand_root(self, root, env):

        priors = self._policy(env)

        root.expand(priors)


    # ==================================================
    # POLICY NETWORK
    # ==================================================

    def _policy(self, env):
        """
        Get NN policy for state

        Returns:
            dict[action_index] = probability
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
            # Uniform fallback
            n = len(priors)

            for k in priors:
                priors[k] = 1.0 / n


        return priors


    # ==================================================
    # VALUE NETWORK
    # ==================================================

    def _evaluate(self, env):
        """
        Value evaluation
        """

        state = env.get_state()

        s = torch.tensor(state).float().unsqueeze(0)


        with torch.no_grad():
            _, value = self.net(s)


        return value.item()
