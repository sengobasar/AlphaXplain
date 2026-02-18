import subprocess


class OllamaExplainer:
    """
    Uses local Ollama LLM to explain AI reasoning
    """

    def __init__(self, model="llama3"):
        self.model = model


    def explain(self, reasoning_data):
        """
        reasoning_data: list of dicts from MCTS

        Returns:
            natural language explanation (str)
        """

        prompt = self._build_prompt(reasoning_data)

        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            return "LLM error: " + result.stderr

        return result.stdout.strip()


    # =========================================


    def _build_prompt(self, data):

        text = """
You are a chess coach.

Explain why the AI chose its move using the statistics below.
Be clear, short, and human-friendly.

Do NOT mention neural networks.
Do NOT mention MCTS.
Explain in chess terms.

Candidate moves:
"""

        for i, d in enumerate(data, 1):

            text += (
                f"{i}. Move: {d['move']} | "
                f"Visits: {d['visits']} | "
                f"Score: {d['Q']} | "
                f"Prior: {d['prior']}\n"
            )

        text += "\nGive the best explanation."

        return text
