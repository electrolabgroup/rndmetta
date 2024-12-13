from transformers import pipeline

# Set your Hugging Face API key
api_key = "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"

# Set up the pipeline with the API key
pipe = pipeline("text-generation", model="meta-llama/CodeLlama-7b-hf", hf_token=api_key)

# Now you can use the pipeline for text generation
prompt = "Write a short story about a friendly AI assistant."
output = pipe(prompt)
print(output)