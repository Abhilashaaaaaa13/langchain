# The typo is fixed here: ChatHuggingFace
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import torch

print("Initializing local model... This may take a long time on the first run.")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device_map="auto",
    pipeline_kwargs=dict(
        max_new_tokens=100,
        temperature=0.5,
    ),
)

model = ChatHuggingFace(llm=llm)

print("Model loaded. Sending request...")
result = model.invoke("What is the capital of India?")

print("\nModel Response:")
print(result.content)