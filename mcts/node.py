import math


class MCTSNode:
    """
    One node in the Monte Carlo Tree Search.
    """

    def __init__(self, parent=None, prior=0.0):

        self.parent = parent

        # Children: move_index -> node
        self.children = {}

        # Visit count
        self.N = 0

        # Total value
        self.W = 0.0

        # Mean value
        self.Q = 0.0

        # Prior probability from policy
        self.P = prior


    def is_leaf(self):
        return len(self.children) == 0


    def expand(self, priors):
        """
        priors: dict {action_index : probability}
        """

        for action, prob in priors.items():

            if action not in self.children:

                self.children[action] = MCTSNode(
                    parent=self,
                    prior=prob
                )


    def best_child(self, c_puct=1.4):
        """
        Select child using PUCT formula
        """

        best_score = -float("inf")
        best_action = None
        best_node = None


        for action, child in self.children.items():

            # UCB / PUCT score
            u = (
                c_puct
                * child.P
                * math.sqrt(self.N + 1e-8)
                / (1 + child.N)
            )

            score = child.Q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_node = child


        return best_action, best_node


    def backup(self, value):
        """
        Backpropagate value up the tree
        """

        node = self

        while node is not None:

            node.N += 1
            node.W += value
            node.Q = node.W / node.N

            # Flip value for opponent
            value = -value

            node = node.parent
