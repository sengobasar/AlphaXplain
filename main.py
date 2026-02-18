import torch

from env.chess_env import ChessEnv
from agent.network import AlphaZeroNet
from utils.move_encoder import MoveEncoder
from mcts.search import MCTS


def main():

    env = ChessEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = AlphaZeroNet().to(device)
    net.eval()

    encoder = MoveEncoder()

    # Create MCTS
    mcts = MCTS(
        net=net,
        encoder=encoder,
        simulations=50,   # increase later
        c_puct=1.4
    )


    episodes = 3


    for ep in range(episodes):

        print(f"\n=== Game {ep} ===\n")

        env.reset()

        done = False
        total_reward = 0


        while not done:

            env.render()

            # -------------------------
            # RUN MCTS
            # -------------------------

            root = mcts.run(env)


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
                print("No action from MCTS!")
                break


            move = encoder.decode(best_action)


            # -------------------------
            # PLAY MOVE
            # -------------------------

            state, reward, done = env.step(move)

            total_reward += reward


        print("\nGame Over")
        print("Final Reward:", total_reward)



if __name__ == "__main__":
    main()
