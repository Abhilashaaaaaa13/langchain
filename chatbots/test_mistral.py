from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

# This script assumes you have already logged in using: hf auth login

print("Initializing the model endpoint...")

# Use the recommended, non-gated model from Mistral AI
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

try:
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=256,
    )

    model = ChatHuggingFace(llm=llm)

    print(f"Successfully initialized model: {repo_id}")
    print("-" * 30)
    print("Sending request to the model...")

    prompt = "What is the capital of India?"
    result = model.invoke(prompt)

    print("\nModel Response:")
    print(result.content)

except Exception as e:
    print("\nAn error occurred:")
    print(e)