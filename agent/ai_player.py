import torch

from env.chess_env import ChessEnv
from agent.network import AlphaZeroNet
from utils.move_encoder import MoveEncoder
from mcts.search import MCTS


class AIPlayer:
    """
    AlphaZero-based Chess AI Player
    """

    def __init__(
        self,
        model_path,
        device=None,
        simulations=50
    ):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = device

        # Load network
        self.net = AlphaZeroNet().to(device)
        self.net.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.net.eval()

        # Encoder
        self.encoder = MoveEncoder()

        # MCTS
        self.mcts = MCTS(
            net=self.net,
            encoder=self.encoder,
            simulations=simulations,
            c_puct=1.4
        )


    def select_move(self, board):

        """
        board: chess.Board object
        Returns: chess.Move
        """

        # Create temp env
        env = ChessEnv()
        env.board = board.copy()

        # Run MCTS
        root = self.mcts.run(env)

        # Pick best visit count
        best_N = -1
        best_action = None

        for action, child in root.children.items():

            if child.N > best_N:
                best_N = child.N
                best_action = action


        if best_action is None:
            return None


        move = self.encoder.decode(best_action)

        return move
