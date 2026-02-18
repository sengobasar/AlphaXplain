import pygame
import sys
import chess

from agent.ai_player import AIPlayer


# -----------------------------
# CONFIG
# -----------------------------

WIDTH = 640
HEIGHT = 700      # Extra space for text

SQ_SIZE = WIDTH // 8

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)

HIGHLIGHT = (186, 202, 68)

FPS = 30


TEXT_COLOR = (30, 30, 30)
TEXT_BG = (220, 220, 220)


# -----------------------------
# LOAD PIECE IMAGES
# -----------------------------

def load_images():

    images = {}

    pieces = ["P", "R", "N", "B", "Q", "K"]

    for p in pieces:

        # White
        w_path = f"visual/pieces/w{p}.png"
        w_img = pygame.image.load(w_path)
        w_img = pygame.transform.scale(w_img, (SQ_SIZE, SQ_SIZE))
        images["w" + p] = w_img

        # Black
        b_path = f"visual/pieces/b{p}.png"
        b_img = pygame.image.load(b_path)
        b_img = pygame.transform.scale(b_img, (SQ_SIZE, SQ_SIZE))
        images["b" + p] = b_img

    return images


# -----------------------------
# DRAW BOARD
# -----------------------------

def draw_board(screen):

    for r in range(8):
        for c in range(8):

            color = LIGHT if (r + c) % 2 == 0 else DARK

            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(
                    c * SQ_SIZE,
                    r * SQ_SIZE,
                    SQ_SIZE,
                    SQ_SIZE
                )
            )


def draw_pieces(screen, board, images):

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece:

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            color = "w" if piece.color == chess.WHITE else "b"
            key = color + piece.symbol().upper()

            screen.blit(
                images[key],
                pygame.Rect(
                    col * SQ_SIZE,
                    row * SQ_SIZE,
                    SQ_SIZE,
                    SQ_SIZE
                )
            )


def highlight_square(screen, square):

    if square is None:
        return

    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)

    pygame.draw.rect(
        screen,
        HIGHLIGHT,
        pygame.Rect(
            col * SQ_SIZE,
            row * SQ_SIZE,
            SQ_SIZE,
            SQ_SIZE
        ),
        4
    )


# -----------------------------
# DRAW EXPLANATION PANEL
# -----------------------------

def draw_explanation(screen, text, font):

    panel_y = 8 * SQ_SIZE
    panel_h = HEIGHT - panel_y

    pygame.draw.rect(
        screen,
        TEXT_BG,
        pygame.Rect(0, panel_y, WIDTH, panel_h)
    )


    if not text:
        return


    lines = wrap_text(text, font, WIDTH - 20)

    y = panel_y + 10


    for line in lines:

        render = font.render(line, True, TEXT_COLOR)

        screen.blit(render, (10, y))

        y += render.get_height() + 4


def wrap_text(text, font, max_width):

    words = text.split(" ")

    lines = []
    current = ""


    for w in words:

        test = current + w + " "

        if font.size(test)[0] <= max_width:

            current = test

        else:

            lines.append(current)
            current = w + " "


    lines.append(current)

    return lines


# -----------------------------
# MAIN GUI
# -----------------------------

def main():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Explainable AlphaZero Chess")

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 18)


    images = load_images()

    board = chess.Board()


    ai = AIPlayer(
        model_path="model_iter_4.pth",
        simulations=50
    )


    selected_square = None

    running = True

    last_explanation = ""


    while running:

        clock.tick(FPS)


        # -------------------------
        # EVENTS
        # -------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False


            if event.type == pygame.MOUSEBUTTONDOWN:

                if board.is_game_over():
                    continue


                x, y = pygame.mouse.get_pos()

                col = x // SQ_SIZE
                row = y // SQ_SIZE

                square = chess.square(col, 7 - row)


                # First click
                if selected_square is None:

                    piece = board.piece_at(square)

                    if piece and piece.color == chess.WHITE:

                        selected_square = square


                # Second click
                else:

                    move = chess.Move(selected_square, square)


                    # Promotion
                    if (
                        board.piece_at(selected_square).piece_type == chess.PAWN
                        and chess.square_rank(square) in [0, 7]
                    ):
                        move.promotion = chess.QUEEN


                    if move in board.legal_moves:

                        board.push(move)

                        selected_square = None


                        # -------------------------
                        # AI MOVE + EXPLANATION
                        # -------------------------

                        if not board.is_game_over():

                            ai_move, explanation = ai.select_move(board)

                            if ai_move:

                                board.push(ai_move)

                                last_explanation = explanation

                                print("\nAI Explanation:\n")
                                print(explanation)
                                print("\n-----------------\n")


                    else:
                        selected_square = None


        # -------------------------
        # DRAW
        # -------------------------

        draw_board(screen)

        highlight_square(screen, selected_square)

        draw_pieces(screen, board, images)

        draw_explanation(screen, last_explanation, font)

        pygame.display.flip()


    pygame.quit()
    sys.exit()



if __name__ == "__main__":
    main()
