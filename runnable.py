from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Step 1: HF pipeline load karo
generator = pipeline("text-generation", model="gpt2", max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=generator)

# Step 2: Prompts banao
prompt1 = PromptTemplate.from_template("Tell me a short joke about {topic}.")
prompt2 = PromptTemplate.from_template("Now explain why this joke is funny: {joke}")

# Step 3: Parser
parser = StrOutputParser()

# Step 4: SequentialRunnable banao
joke_chain = RunnableSequence(first=prompt1, middle=[llm, parser])
explain_chain = RunnableSequence(first=prompt2, middle=[llm, parser])

# Step 5: Pehle joke generate
joke = joke_chain.invoke({"topic": "dogs"})
print("🃏 Joke:", joke)

# Step 6: Fir explanation us joke ka
explanation = explain_chain.invoke({"joke": joke})
print("\n📖 Explanation:", explanation)
