from flask import Flask, request, jsonify, render_template

from transformers import pipeline
import torch

app = Flask(__name__)

# Hugging Face token for authentication (use your token here)
HF_TOKEN = "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"

# Load the CodeLlama model using the Hugging Face pipeline API
model_name = "meta-llama/CodeLlama-7b-hf"
generator = pipeline("text-generation", model=model_name, use_auth_token=HF_TOKEN)

BATCH_SIZE = 8  # Define the batch size for batching prompts


def process_batch(prompts):
    """
    Process a batch of prompts using the model.
    """
    results = generator(prompts, max_length=200, num_return_sequences=1, temperature=0.7)
    return [result['generated_text'] for result in results]


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/chat_batch", methods=["POST"])
def chat_batch():
    """
    Handle batch prompts and generate responses.
    """
    try:
        data = request.json
        prompts = data.get("prompts", [])

        if not prompts or not isinstance(prompts, list):
            return jsonify({"error": "No prompts provided or invalid format"}), 400

        responses = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_responses = process_batch(batch_prompts)
            responses.extend(batch_responses)

        return jsonify({"responses": responses})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
