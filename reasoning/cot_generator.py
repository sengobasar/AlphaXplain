def explain(state, action, value, reward):

    text = f"""
STATE: {state}

I estimated value V(s) ≈ {value:.3f}

I chose action: {action}

I received reward: {reward}

If reward is low → update policy.
If reward is high → reinforce action.

Learning continues.
"""

    return text
