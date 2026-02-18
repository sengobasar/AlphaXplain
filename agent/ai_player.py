import torch

from env.chess_env import ChessEnv
from agent.network import AlphaZeroNet
from utils.move_encoder import MoveEncoder
from mcts.search import MCTS

from llm.explainer import OllamaExplainer


class AIPlayer:
    """
    AlphaZero-based Chess AI Player
    + LLM Explanation Layer
    """

    def __init__(
        self,
        model_path,
        device=None,
        simulations=10,
        llm_model="phi"
    ):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = device


        # -------------------------
        # Neural Network
        # -------------------------

        self.net = AlphaZeroNet().to(device)

        self.net.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        self.net.eval()


        # -------------------------
        # Encoder
        # -------------------------

        self.encoder = MoveEncoder()


        # -------------------------
        # MCTS
        # -------------------------

        self.mcts = MCTS(
            net=self.net,
            encoder=self.encoder,
            simulations=simulations,
            c_puct=1.4
        )


        # -------------------------
        # LLM Explainer
        # -------------------------

        self.explainer = OllamaExplainer(model=llm_model)



    # ==================================================
    # MAIN MOVE FUNCTION
    # ==================================================

    def select_move(self, board):
        """
        board: chess.Board

        Returns:
            move (chess.Move)
            explanation (str)
        """

        # Create temporary environment
        env = ChessEnv()
        env.board = board.copy()


        # -------------------------
        # RUN MCTS
        # -------------------------

        root, reasoning = self.mcts.run(env)


        # -------------------------
        # SELECT BEST MOVE
        # -------------------------

        best_N = -1
        best_action = None


        for action, child in root.children.items():

            if child.N > best_N:
                best_N = child.N
                best_action = action


        if best_action is None:
            return None, "No valid move found."


        move = self.encoder.decode(best_action)


        # -------------------------
        # EXPLAIN WITH LLM
        # -------------------------

        explanation = self._explain(reasoning)


        return move, explanation



    # ==================================================
    # LLM EXPLANATION
    # ==================================================

    def _explain(self, reasoning):

        try:

            text = self.explainer.explain(reasoning)

            if text.strip() == "":
                return "No explanation generated."

            return text


        except Exception as e:

            return f"Explanation error: {e}"
