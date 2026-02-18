import torch

from env.chess_env import ChessEnv
from agent.network import AlphaZeroNet
from utils.move_encoder import MoveEncoder
from mcts.search import MCTS
from rl.replay_buffer import ReplayBuffer
from rl.trainer import AlphaZeroTrainer


def main():

    env = ChessEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Network
    net = AlphaZeroNet().to(device)

    # Encoder
    encoder = MoveEncoder()

    # Memory
    buffer = ReplayBuffer(capacity=10000)

    # Trainer
    trainer = AlphaZeroTrainer(net, lr=1e-3)

    # MCTS
    mcts = MCTS(
        net=net,
        encoder=encoder,
        simulations=50,
        c_puct=1.4
    )


    # -------------------------
    # TRAINING LOOP
    # -------------------------

    iterations = 5          # learning cycles
    games_per_iter = 10     # self-play games
    train_epochs = 2


    for it in range(iterations):

        print(f"\n============================")
        print(f" ITERATION {it}")
        print(f"============================\n")


        # -------------------------
        # SELF-PLAY
        # -------------------------

        net.eval()

        for ep in range(games_per_iter):

            print(f"\nSelf-Play Game {ep}")

            env.reset()

            done = False

            game_data = []


            while not done:

                # Get state
                state = env.get_state()


                # Run MCTS
                root = mcts.run(env)


                # Build pi
                pi = torch.zeros(encoder.size)

                total_visits = 0


                for action, child in root.children.items():

                    pi[action] = child.N
                    total_visits += child.N


                if total_visits > 0:
                    pi = pi / total_visits
                else:
                    pi = pi + 1.0 / encoder.size


                # Store state + pi
                game_data.append((state, pi.numpy()))


                # Select move
                action = torch.argmax(pi).item()

                move = encoder.decode(action)


                # Play
                _, reward, done = env.step(move)


            # -------------------------
            # GAME RESULT
            # -------------------------

            if reward == 1:
                z = 1
            elif reward == -1:
                z = -1
            else:
                z = 0


            # Store in buffer
            for state, pi in game_data:

                buffer.add(state, pi, z)


            print("Game result:", z)
            print("Buffer size:", len(buffer))


        # -------------------------
        # TRAINING
        # -------------------------

        print("\nTraining...")

        trainer.train(
            buffer,
            batch_size=64,
            epochs=train_epochs
        )


        # -------------------------
        # SAVE MODEL
        # -------------------------

        torch.save(
            net.state_dict(),
            f"model_iter_{it}.pth"
        )

        print(f"Model saved: model_iter_{it}.pth")


    print("\nTraining complete!")



if __name__ == "__main__":
    main()
