# ♟️ AlphaXplain — Explainable AlphaZero Chess Engine

> *AlphaZero + Explainability. A chess engine that doesn't just play — it thinks out loud.*

```
AlphaXplain = AlphaZero-style Self-Play RL + MCTS + Structured LLM Reasoning
```
<img width="1917" height="1011" alt="image" src="https://github.com/user-attachments/assets/98e6715c-1548-4106-9be1-1d6b4b6acd3e" />



---

## ⚠️ Honest Disclaimer

> **Due to limited compute, I trained for ~30 iterations (~0.5 hours on a consumer CPU/GPU).**
> Full-scale AlphaZero required thousands of Google TPUs running for weeks.
>
> **This means the engine plays terribly — and that's completely expected.**
> The goal of AlphaXplain is not superhuman chess. It's to demonstrate the *algorithmic architecture* of a self-learning, self-explaining AI system in a resource-efficient, fully local environment.
>
> Think of it as AlphaZero's brain transplanted into a laptop. The wiring is the same. The muscles just haven't trained yet.

---

## 🧠 What Is AlphaXplain?

AlphaXplain is a chess-playing AI that learns entirely from self-play — no human games, no handcrafted rules — and then explains its own reasoning in plain English using a local language model.

I built it on three ideas that rarely meet in one project:

1. **AlphaZero-style reinforcement learning** — the engine teaches itself by playing against itself, guided only by wins and losses.
2. **Monte Carlo Tree Search (MCTS)** — instead of brute-force calculation, it explores the future intelligently, using its neural network as a guide.
3. **Structured LLM Explanation** — after each move, MCTS statistics (visit counts, Q-values, policy probabilities) are passed to a local language model (Phi via Ollama), which converts raw numbers into human-readable reasoning.

This combination produces what is rarely seen in game-playing AI: **an engine that can explain why it made a decision.**

---

## 🟡 Novel Contribution: Grounded Chain-of-Thought Extraction

Most chess engines are black boxes. You see the move — not the reasoning.

AlphaXplain introduces a **structured reasoning extraction layer**: MCTS statistics are serialized into a structured format and passed to a local LLM (Phi via Ollama). The model generates natural language explanation grounded in actual search data — not hallucination. This is what researchers call a **faithful explanation pipeline**.

To my knowledge, few publicly available implementations integrate structured MCTS statistics with a local language model to produce grounded, move-level natural language explanations.

**Example output:**
```
Move played: e7e6
Score: -0.011 | Visits: 3

AlphaXplain:
"e7e6 was selected because it supports central control and
 avoids immediate tactical threats. The policy network
 assigned it the highest prior, and MCTS confirmed this
 with the most visits during search."
```

---

## 📐 Mathematical Foundations

Every equation below maps directly to a module in the codebase.

---

### 1. Markov Decision Process (MDP)

The chess environment is formally defined as:

```
(S, A, P, R, γ)
```

| Symbol | Meaning | Chess Mapping |
|---|---|---|
| `S` | State space | All legal board positions |
| `A` | Action space | All legal moves |
| `P` | Transition function | Deterministic move outcomes |
| `R` | Reward function | +1 win, 0 draw, -1 loss |
| `γ` | Discount factor | 0.99 — values near-future outcomes |

Chess is a **deterministic, fully observable, two-player zero-sum MDP** — one of the cleanest environments RL can operate in.

---

### 2. State Encoding

Raw board positions cannot be fed directly to a neural network. I apply an encoding function:

```
φ: S → ℝ^(8×8×12)
```

The 8×8 board is encoded as 12 binary planes — one per piece type per color (6 piece types × 2 colors). Additional planes encode castling rights, turn, and move count. This gives the network spatially structured input that convolutions can process.

---

### 3. Action Encoding

All chess moves are mapped to a structured action space:

```
ψ: A → ℝ^(73×8×8)
```

73 move planes × 64 squares = 4,672 possible actions. Each plane encodes a specific move type (queen moves by direction, knight moves, underpromotions). This lets the policy head output a full distribution over all legal moves in a single forward pass.

---

### 4. Neural Network — Policy and Value

The policy and value heads share a deep residual backbone:

```
(p, v) = f_θ(s)
```

| Symbol | Meaning |
|---|---|
| `s` | Encoded board state |
| `f_θ` | Neural network with learnable parameters θ |
| `p` | Policy vector — probability distribution over all moves |
| `v` | Value scalar ∈ [-1, +1] — predicted game outcome |

#### Convolution (Spatial Pattern Detection)

```
y(i,j) = Σ_{u,v} x(i+u, j+v) · k(u,v)
```

Convolutional filters detect spatial patterns — pins, forks, open files — by sliding learned kernels across the board representation.

#### Residual Block (Deep Network Stability)

```
y = F(x) + x
```

Skip connections allow gradients to flow cleanly through deep networks, preventing vanishing gradients. AlphaZero uses 20–40 residual blocks. AlphaXplain uses a reduced version appropriate to available compute.

---

### 5. Monte Carlo Tree Search (MCTS)

MCTS is the search backbone. It builds a game tree guided by the neural network, accumulating visit statistics to identify the strongest moves.

#### PUCT Formula — The Heart of AlphaXplain

At each node, the move to explore is selected by:

```
a = argmax_a ( Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)) )
```

| Symbol | Meaning |
|---|---|
| `Q(s,a)` | Mean value estimate for move `a` from state `s` |
| `P(s,a)` | Prior probability from the neural network |
| `N(s)` | Total visits to state `s` |
| `N(s,a)` | Visits to move `a` from state `s` |
| `c_puct` | Exploration constant (typically 1.0–2.5) |

This formula elegantly balances **exploitation** (high Q — moves that worked) against **exploration** (high P, low N — promising but untested moves). As `N(s,a)` grows, the exploration bonus shrinks, naturally focusing search on the best lines.

#### MCTS Backup Equations

After each simulation reaches a terminal or leaf node, statistics are propagated backwards through the tree:

```
N(s,a) ← N(s,a) + 1
W(s,a) ← W(s,a) + v
Q(s,a) = W(s,a) / N(s,a)
```

`W` accumulates total value; `Q` is its running mean. Over many simulations, `Q` converges to an accurate estimate of move quality — and these are the exact values AlphaXplain passes to the LLM for explanation.

---

### 6. Self-Play and Approximate Policy Iteration

AlphaXplain's learning process is a form of **Approximate Policy Iteration (API)**:

```
π_{t+1} ≈ MCTS(π_t)
```

Each generation's policy drives self-play. MCTS improves upon the raw network policy `p` to produce a stronger search policy `π`. The network then learns to imitate `π`, creating a virtuous cycle:

```
Policy → MCTS Search → Better Policy → Train Network → Stronger Policy → Repeat
```

The training dataset is formally defined as:

```
D = { (s_t, π_t, z) }
```

Each sample is a (board state, MCTS policy, game outcome) triple. This is **supervised learning on self-generated data** — no human knowledge required at any stage.

---

### 7. Value Function and Bellman Backup

The value function estimates expected return from a state:

```
V(s) = E[ Σ_{t=0}^{∞} γ^t r_t ]
```

The **Bellman backup** — the theoretical backbone of all reinforcement learning — connects values across time steps:

```
V(s) = E[ r + γ V(s') ]
```

For action-values:

```
Q(s,a) = r + γ · max_a Q(s', a)
```

This recurrence is what allows AlphaXplain to reason about the future. Every value prediction is anchored to actual game outcomes through this chain.

---

### 8. Policy Gradient Theory

The policy improves via the **Policy Gradient Theorem**:

```
∇_θ J = E[ ∇_θ log π(a|s;θ) · A(s,a) ]
```

Where the **advantage function** is:

```
A(s,a) = Q(s,a) - V(s)
```

The advantage answers: *was this move better or worse than average in this position?* Positive advantage increases the move's probability; negative decreases it. This is not magic — it is the direction of the gradient.

---

### 9. Loss Function

The network minimizes a combined loss over the training dataset:

```
l = (z - v)² - πᵀ log p + c‖θ‖²
```

| Term | Name | Meaning |
|---|---|---|
| `(z - v)²` | Value loss | How wrong was the win prediction? |
| `-πᵀ log p` | Policy loss (cross-entropy) | How different is `p` from the MCTS-refined policy `π`? |
| `c‖θ‖²` | L2 regularization | Prevent overfitting, keep weights bounded |

The cross-entropy term is the engine of self-teaching: the network learns to predict what MCTS recommends, compressing tree search knowledge directly into network weights.

---

### 10. Optimization — Adam

Weights are updated via gradient descent:

```
θ ← θ - α ∇_θ l
```

In practice, I use **Adam**, which maintains adaptive moment estimates for stable training:

```
m_t = β₁ m_{t-1} + (1 - β₁) g_t          # First moment (mean)
v_t = β₂ v_{t-1} + (1 - β₂) g_t²         # Second moment (variance)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)   # Bias-corrected update
```

Adam prevents oscillation during training and handles sparse gradients well — critical when most board positions appear only once during early self-play.

---

### 11. Exploration — Dirichlet Noise

To prevent the root node from collapsing to a single line, I inject noise at the start of each search:

```
P'(s,a) = (1 - ε) P(s,a) + ε Dir(α)
```

| Symbol | Meaning |
|---|---|
| `ε` | Noise weight (typically 0.25) |
| `Dir(α)` | Dirichlet distribution draw (α ≈ 0.3 for chess) |

Without this, the policy becomes deterministic too early — a failure mode called **policy collapse**. Dirichlet noise ensures **ergodicity**: every move retains a non-zero probability of being explored, preventing the engine from getting locked into a narrow repertoire.

---

### 12. Convergence and Limitations

AlphaXplain approximates the optimal policy:

```
π* = argmax_π J(π)
```

Convergence is theoretically guaranteed under infinite compute and perfect function approximation. At this scale, real limitations apply: function approximation error from a small network, insufficient self-play data, distribution shift between training phases, and short MCTS simulation budgets. These are expected consequences of the compute gap — not architectural flaws.

---

### 13. Computational Complexity

| Component | Complexity |
|---|---|
| MCTS per move | O(simulations × depth) |
| Network forward pass | O(parameters × board size) |
| Training step | O(batch size × parameters) |
| Full training run | O(iterations × games × moves per game) |

At full AlphaZero scale these numbers become astronomical. AlphaXplain makes deliberate trade-offs at every level to remain runnable on a single consumer machine.

---

## 🔁 The Full AlphaXplain Loop

```
THINK      (p, v) = f_θ(s)                     Network evaluates position
ACT        a ~ π(a|s)                           Sample move from MCTS policy
OBSERVE    (s_t, a_t, r_t, s_{t+1}, z)         Record experience
LEARN      l = (z-v)² - πᵀ log p + c‖θ‖²       Compute combined loss
IMPROVE    θ ← θ - α∇_θ l                       Update weights
EXPLAIN    MCTS stats → LLM → natural language  AlphaXplain's unique step
REPEAT     θ₀ → θ₁ → θ₂ → ...                  Each generation stronger
```

---

## 🚀 Future Work

### 🟢 1. Coach Mode
After a game ends, pass the full PGN to the LLM:
> *"Explain my biggest mistake."*

### 🟢 2. Opening Trainer
Auto-detect the opening (Sicilian, French, Ruy López) and explain the strategic plan behind each.

### 🟢 3. Blunder Detector
If Q-value drops sharply after a move:
> *"You blundered because your knight on e5 became undefended."*

### 🟢 4. Chat With AlphaXplain
```
User:          Why didn't you castle?
AlphaXplain:   Castling was deprioritized because the kingside pawns
               were advanced, reducing shelter. MCTS visited it only
               2 times vs 18 for the chosen move.
```

### 🟢 5. Commentary Mode
Real-time narration during play:
> *"White increases pressure on the d-file. Black's queen is running out of squares..."*

---

## 📁 Project Structure

```
alphaxplain/
│
├── agent/          # AI agents and player wrappers
├── env/            # Chess environment (python-chess)
├── mcts/           # Monte Carlo Tree Search + PUCT
├── rl/             # Self-play and training loop
├── llm/            # LLM explanation generator (Ollama + Phi)
├── utils/          # Helper functions and state encoders
├── visual/         # Pygame GUI and board rendering
├── models/         # Saved model checkpoints (.pth)
└── main.py         # Training entry point
```

Each module is fully isolated. You can swap the neural network, LLM, or search algorithm independently.

---

## 🛠️ Tools and Technologies

| Component | Tool |
|---|---|
| Game Logic | `python-chess` |
| Neural Networks | `PyTorch` |
| GUI | `Pygame` |
| Local LLM | `Ollama` + `Phi` |
| Numerics | `NumPy` |
| Language | `Python 3.12` |

**Everything runs locally. No cloud APIs. No internet required after setup.**

---

## ▶️ Running AlphaXplain

```bash
# Launch the visual chess GUI
python -m visual.chess_gui
```

AlphaXplain will play against you and explain each move in the terminal.

---

## 📊 Reality Check

| | AlphaXplain | Original AlphaZero |
|---|---|---|
| Hardware | 1× consumer CPU/GPU | Thousands of TPUs |
| Training Time | ~30 minutes | Weeks |
| Iterations | ~30 | Millions |
| Playing Strength | Beginner | Superhuman |
| Architecture | Same ✅ | Same ✅ |
| Explainability | ✅ Yes | ❌ No |

The gap in strength is compute. The architecture is identical in spirit. And AlphaXplain does something the original never did: **it explains itself.**

---

## 🔍 System Overview

**Environment** — Built on `python-chess`. Handles board representation, legal move generation, game termination, and state encoding as φ: S → ℝ^(8×8×12).

**Neural Network** — A deep residual network that outputs a policy (move probabilities) and a value (position evaluation). Trained exclusively from self-play — no human games, no opening books.

**Monte Carlo Tree Search** — Explores future positions using the PUCT formula. Visit statistics are backed up after each simulation and serve as both the move selection mechanism and the raw material for explanation.

**Reinforcement Learning Loop** — Self-play generates dataset D = {(s_t, π_t, z)}. The network trains on this data. The model updates. The loop repeats.

**LLM Explanation Layer** — MCTS outputs (Q-values, visit counts, policy priors) are extracted as structured data and passed to a local language model. The model generates faithful natural language reasoning grounded in real search statistics — AlphaXplain's core contribution.

---

## 🧩 Why AlphaXplain Matters

Explainable AI (XAI) is one of the most important open problems in machine learning. Most powerful models — including the original AlphaZero — are black boxes. You see the output, not the reasoning.

AlphaXplain is a working prototype that bridges that gap: a reinforcement learning engine that extracts its own reasoning from tree search statistics and communicates it in natural language. With stronger compute and training, this architecture scales directly toward a genuinely interpretable game-playing agent.

That combination — **RL + MCTS + grounded LLM reasoning** — is the contribution.

---

## 👤 Author

Built alone, with curiosity, constrained compute, and the belief that understanding *how* an AI thinks matters as much as *how well* it plays.

---

*"The goal was never to beat Stockfish. The goal was to build something that can tell you why it moved."*

