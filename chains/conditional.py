from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')
parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template="classify the sentiment of the follwing feedback text into positive or negative \n {feedback} \n {format_instruction} ",
    input_variables= ['feedback'],
    partial_variables= {'format_instruction':parser2.get_format_instructions()}
)
classifier_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(
    template="Write an appropriate esponse to this positive feedback \n {feedback} ",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Write an appropriate esponse to this negative feedback \n {feedback} ",
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x['sentiment']=='positive',prompt2|model| parser),
    (lambda x:x['sentiment']=='negative',prompt2|model| parser),
    RunnableLambda(lambda x: "couldnot find sentiment")
   

)
chain = classifier_chain | branch_chain
print(chain.invoke({'feedback': 'This is a beautiful phone'}))