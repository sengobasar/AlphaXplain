import torch
import torch.nn.functional as F
import torch.optim as optim


class AlphaZeroTrainer:
    """
    Trains network using AlphaZero loss:

    L = (z - v)^2  -  pi * log(p)
    """

    def __init__(self, net, lr=1e-3, weight_decay=1e-4):

        self.net = net

        self.optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


    def train(self, buffer, batch_size=64, epochs=1):

        if len(buffer) < batch_size:
            print("Not enough data to train yet.")
            return


        self.net.train()


        for epoch in range(epochs):

            states, target_policies, target_values = buffer.sample(batch_size)

            # Move to device
            device = next(self.net.parameters()).device

            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)


            # Forward
            pred_policies, pred_values = self.net(states)

            pred_values = pred_values.squeeze()


            # -------------------------
            # POLICY LOSS
            # -------------------------

            log_probs = torch.log(pred_policies + 1e-8)

            policy_loss = -torch.mean(
                torch.sum(target_policies * log_probs, dim=1)
            )


            # -------------------------
            # VALUE LOSS
            # -------------------------

            value_loss = F.mse_loss(pred_values, target_values)


            # -------------------------
            # TOTAL LOSS
            # -------------------------

            loss = policy_loss + value_loss


            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            print(
                f"Epoch {epoch} | "
                f"Loss: {loss.item():.4f} | "
                f"Policy: {policy_loss.item():.4f} | "
                f"Value: {value_loss.item():.4f}"
            )


        self.net.eval()
