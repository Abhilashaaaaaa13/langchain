import json
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

# Define output schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either positive, negative or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

# Setup LLaMA-3 model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3-8b-instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# Review text
review_text = """The hardware is great, but then the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this."""

# Strict JSON prompt
prompt = f"""
You are a JSON-only data extractor.

Extract the following fields from the review:
- key_themes: list of key topics
- summary: short summary
- sentiment: "positive", "negative", or "neutral"
- pros: list of pros
- cons: list of cons

Output ONLY valid JSON, no extra text, no explanations.
JSON must strictly match this schema:
{{
  "key_themes": ["string", ...],
  "summary": "string",
  "sentiment": "string",
  "pros": ["string", ...] or null,
  "cons": ["string", ...] or null
}}

Review: "{review_text}"
"""

# Get model output
response = model.invoke(prompt)
raw_output = getattr(response, "content", str(response)).strip()

# Try parsing JSON
try:
    parsed: Review = json.loads(raw_output)
except json.JSONDecodeError:
    print("Invalid JSON output. Raw output:\n", raw_output)
    parsed = None

print(parsed)
