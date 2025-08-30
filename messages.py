from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation"
    )
model = ChatHuggingFace(llm=llm)
messages = [
    SystemMessage(content = 'Your are a helpful assisstant'),
    HumanMessage(content='Tell me about Langchain')
]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)