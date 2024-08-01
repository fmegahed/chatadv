
# Based on https://python.langchain.com/v0.1/docs/use_cases/question_answering/sources/

# Directory to store embeddings
EMBEDDINGS_DIRECTORY = './vstore/'

# -----------------------------------------------------------------------------
# Importing the Required Libraries:
# ---------------------------------
import os
import pickle

# third-party libraries
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate
)

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
# -----------------------------------------------------------------------------


# Loading the environmental variables:
# ------------------------------------
load_dotenv(find_dotenv(), override=True)
open_api_key = os.environ.get('OPENAI_API_KEY')
# -----------------------------------------------------------------------------


# Loading the data, embeddings, and vectorstore:
# ----------------------------------------------
embeddings_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
vectorstore = FAISS.load_local(
  'vstore/webpage_vectorstore', 
  embeddings = embeddings_model, 
  allow_dangerous_deserialization = True
  )

# Load data from data/website_data.pkl
docs = pickle.load(open('data/website_data.pkl', 'rb'))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# -----------------------------------------------------------------------------  


# The RAG chain:
# --------------
retriever = vectorstore.as_retriever()

system_temp = (
    "You are a friendly chatbot designed to assist Farmer School of Business (FSB) students with their advising questions. "
    "You have access to documents containing information about FSB policies and procedures. Your goal is to answer students' questions using these documents.\n\n"
    
    "Question: {question}\n\n"
    "Context: {context}\n\n"
 
    
    "Instructions for Answering Questions:\n"
    "1. Identify the most relevant quotes from the documents to answer the question.\n"
    "2. List these quotes in numbered order within <quotes></quotes> tags, keeping them short. If no relevant quotes are found, write 'No relevant quotes'.\n"
    "3. Provide your answer inside <answer></answer> tags, referencing the quote numbers in brackets at the end of sentences that draw on those quotes.\n\n"
    
    "Your answer should follow this format, as shown between the <examples></examples> tags:\n\n"
    
    "<examples>\n"
    "<example>\n"
    "What are the required courses for the Finance major?\n"
    "<quotes>\n"
    "[1] Program Requirements\n"
    "[1] ACC 321 Intermediate Financial Accounting 3\n"
    "[1] ECO 301 Money and Banking 3\n"
    "[1] FIN 303 Financial Principles and Introduction to Modeling with Excel 3\n"
    "[1] FIN 381 Intermediate Financial Management 3\n"
    "[1] FIN 401 Principles of Investments and Security Markets 3\n"
    "</quotes>\n"
    "<answer>\n"
    "There are five required courses for Finance majors:[1]\n"
    "  - ACC 321: Intermediate Financial Accounting\n"
    "  - ECO 301: Money and Banking\n"
    "  - FIN 303: Financial Principles and Introduction to Modeling with Excel\n"
    "  - FIN 381: Intermediate Financial Management\n"
    "  - FIN 401: Principles of Investments and Security Markets \n"
    "Additionally, students must complete the required business core courses [2] and select 9 credit hours of finance electives (excluding Capstone Experience courses) [1].\n"
    "ChatAdv encourages consulting the references to verify this information and scheduling a follow-up meeting with your academic advisor. See https://miamioh.edu/fsb/student-resources/academic-advising/appointments.html for more details. ChatAdv is an AI tool and not a replacement for personalized advising.\n"
    "Sources:\n"
    "[1] https://bulletin.miamioh.edu/farmer-business/finance-bsb/\n"
    "[2] https://bulletin.miamioh.edu/farmer-business/\n"
    "</answer>\n"
    "</example>\n"
    
    "<example>\n"
    "What are the prerequisites for ISA 401?\n"
    "<quotes>\n"
    "[1] Prerequisite: ISA 245 or CSE 385.\n"
    "</quotes>\n"
    "<answer>\n"
    "The prerequisite for ISA 401 is ISA 245 or CSE 385 [1].\n"
    "ChatAdv encourages consulting the references to verify this information and scheduling a follow-up meeting with your academic advisor. See https://miamioh.edu/fsb/student-resources/academic-advising/appointments.html for more details. ChatAdv is an AI tool and not a replacement for personalized advising.\n"
    "Source: [1] https://bulletin.miamioh.edu/courses-instruction/isa/\n"
    "</answer>\n"
    "</example>\n"
    "</examples>\n\n"
    
    "If a student's question cannot be sufficiently answered using the provided context, state that directly.\n\n"
    
    "Common terms and abbreviations used by students:\n"
    "prereq = prerequisite\n"
    "prereqs = prerequisites\n"
    "BS = Bachelor of Science\n"
    "BA = Bachelor of Arts\n"
    "Dropping a class = withdrawing from a class\n"
    "Freshman Forgiveness = Course Repeat Policy\n"
    "Pass/Fail = Credit/No Credit\n"
    "DAR = Degree Audit Report\n"
    "Program = Can be a Major, Minor, Co-Major, Certificate, graduate degree, or a Thematic sequence\n"
    "CRN = Course Registration Number\n"
    "ROR = Registration Override Request\n\n"
    
    "Your goal is to be a friendly, knowledgeable resource to help guide students. Always encourage them to verify information and get tailored advice from their assigned academic advisor. Provide relevant information from the context and reference quote numbers and URLs in your answers."
)

prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template=system_temp
            )
        )
    ]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

chat_response = rag_chain_with_source.invoke("What are the graduation requirements for a BS in Business Analytics?")

# extracting the answer and keeping only the text between <answer> and </answer>
chat_answer = chat_response['answer']
chat_answer = chat_answer.split("<answer>")[1].split("</answer>")[0]
print(chat_answer)
