import chess
import numpy as np


class ChessEnv:
    """
    Chess Environment
    """

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        """
        Convert board to 8x8x12 tensor
        (6 piece types × 2 colors)
        """

        state = np.zeros((8, 8, 12), dtype=np.float32)

        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8

            piece_type = piece.piece_type - 1  # 0-5
            color_offset = 0 if piece.color == chess.WHITE else 6

            state[row, col, piece_type + color_offset] = 1

        return state

    def step(self, move):
        self.board.push(move)

        done = self.board.is_game_over()
        reward = 0

        if done:
            result = self.board.result()

            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0

        return self.get_state(), reward, done

    def legal_moves(self):
        return list(self.board.legal_moves)

    def render(self):
        print(self.board)
