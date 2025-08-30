from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm) 
prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt,model,parser)
parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})
final_chain = RunnableSequence(joke_gen_chain)
result = final_chain.invoke({'topic':'AI'})