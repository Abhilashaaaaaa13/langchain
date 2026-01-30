from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model = ChatHuggingFace()
prompt = PromptTemplate(
    template = 'Write a summary for the following poem -\n {poem}',
    input_variables=['poem'] 
    
)
parser = StrOutputParser()
loader = TextLoader('cricket.txt',encoding='utf-8')

docs = loader.load()
print(docs)
print(docs[0])
chain = prompt | model | parser
print( chain.invoke({'poem':docs[0].page_content}))