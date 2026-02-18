import chess


class MoveEncoder:
    """
    Maps chess.Move <-> integer index

    Supports:
    - Normal moves
    - Promotions
    - Castling
    """

    def __init__(self):

        self.move_to_index = {}
        self.index_to_move = {}

        self._build_table()

    def _build_table(self):
        """
        Pre-generate all possible moves (UCI format)
        """

        index = 0

        files = ["a","b","c","d","e","f","g","h"]
        ranks = ["1","2","3","4","5","6","7","8"]

        squares = [f + r for f in files for r in ranks]

        promotions = ["q", "r", "b", "n"]

        for src in squares:
            for dst in squares:

                # Normal move
                move = src + dst

                self.move_to_index[move] = index
                self.index_to_move[index] = move
                index += 1

                # Promotion
                for p in promotions:
                    promo = src + dst + p

                    self.move_to_index[promo] = index
                    self.index_to_move[index] = promo
                    index += 1

        self.size = index

        print("Move space size:", self.size)


    def encode(self, move: chess.Move):
        """
        chess.Move -> index
        """

        uci = move.uci()

        return self.move_to_index.get(uci, None)


    def decode(self, index: int):
        """
        index -> chess.Move
        """

        uci = self.index_to_move.get(index, None)

        if uci is None:
            return None

        return chess.Move.from_uci(uci)
