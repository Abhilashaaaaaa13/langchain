from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()
istral = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
)

falcon = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct"
)
model1 = ChatHuggingFace(istral)
model2 = ChatHuggingFace(falcon)
prompt1 = PromptTemplate(
    template='generate short and simple from the following text {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='generate 5 short questionanswer of the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into a single document \n notes -> {notes} and quiz-> {quiz}',
    input_variables= ['notes','quiz']
    input_variables=['text']
)
parser = StrOutputParser()
paralle_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser
chain = paralle_chain | merge_chain
result = chain.invoke ({'text':'animals'})
print(result)