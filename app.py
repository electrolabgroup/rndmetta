from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

# Hugging Face token for authentication (use your token here)
HF_TOKEN = "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"

# Load the CodeLlama model using the Hugging Face pipeline API
model_name = "meta-llama/CodeLlama-7b-hf"

# Initialize the generator pipeline with FP16 for reduced memory usage
generator = pipeline("text-generation", model=model_name, use_auth_token=HF_TOKEN, device=0, torch_dtype=torch.float16)

BATCH_SIZE = 8  # Define the batch size for batching prompts (reduced for memory)


def process_batch(prompts):
    """
    Process a batch of prompts using the model.
    """
    try:
        results = generator(prompts, max_length=100, num_return_sequences=1, temperature=0.7)
        return [result['generated_text'] for result in results]
    except Exception as e:
        return {"error": f"Error processing batch: {str(e)}"}


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
        # Get JSON data from the request
        data = request.json
        prompts = data.get("prompts", [])

        # Validate prompts
        if not prompts or not isinstance(prompts, list):
            return jsonify({"error": "No prompts provided or invalid format"}), 400

        responses = []
        for i in range(0, len(prompts), BATCH_SIZE):
            # Create batch and process it
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_responses = process_batch(batch_prompts)

            # Handle batch responses, append errors if any
            if "error" in batch_responses:
                return jsonify(batch_responses), 500
            responses.extend(batch_responses)

        return jsonify({"responses": responses})

    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
