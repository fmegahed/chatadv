# Libraries
# ---------
import os
import pickle
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from fpdf import FPDF
import re
import tempfile
from datetime import datetime

def load_environment():
    load_dotenv(find_dotenv(), override=True)
    return os.environ.get('OPENAI_API_KEY')

def load_embeddings_and_vectorstore():
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.load_local(
        'vstore/webpage_vectorstore',
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True
    )
    return embeddings_model, vectorstore

def load_docs():
    return pickle.load(open('data/website_data.pkl', 'rb'))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain():
    _, vectorstore = load_embeddings_and_vectorstore()
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
      "\n\nChatAdv encourages consulting the references to verify this information and scheduling a follow-up meeting with your academic advisor. See https://miamioh.edu/fsb/student-resources/academic-advising/appointments.html for more details. ChatAdv is an AI tool and not a replacement for personalized advising.\n"
      "\n\n**Sources:**\n"
      "\n[1] https://bulletin.miamioh.edu/farmer-business/finance-bsb/\n"
      "\n[2] https://bulletin.miamioh.edu/farmer-business/\n"
      "</answer>\n"
      "</example>\n"
      
      "<example>\n"
      "What are the prerequisites for ISA 401?\n"
      "<quotes>\n"
      "[1] Prerequisite: ISA 245 or CSE 385.\n"
      "</quotes>\n"
      "<answer>\n"
      "The prerequisite for ISA 401 is ISA 245 or CSE 385 [1].\n"
      "\n\nChatAdv encourages consulting the references to verify this information and scheduling a follow-up meeting with your academic advisor. See https://miamioh.edu/fsb/student-resources/academic-advising/appointments.html for more details. ChatAdv is an AI tool and not a replacement for personalized advising.\n"
      "\n\n**Source:** \n[1] https://bulletin.miamioh.edu/courses-instruction/isa/\n"
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

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

# PDF generation functions
def clean_text(input_text):
    LATIN_REPLACEMENTS = {
        "\u2014": "--",     # em dash
        "\u2013": "-",      # en dash
        "\u2018": "'",      # left single quotation mark
        "\u2019": "'",      # right single quotation mark
        "\u201C": "\"",     # left double quotation mark
        "\u201D": "\"",     # right double quotation mark
        "\u2026": "...",    # ellipsis
        "\u00A0": " ",      # non-breaking space
        "\U0001f60a": ":)", # smiling face emoji
    }
    for original, replacement in LATIN_REPLACEMENTS.items():
        input_text = input_text.replace(original, replacement)
    return input_text.encode("latin-1", "ignore").decode("latin-1")

class PDF(FPDF):
    def __init__(self, user_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_name = user_name
        self.date = datetime.now().strftime("%b %d, %Y")
        self.margin = 10

    def header(self):
        self.set_y(8)
        self.set_font("Arial", size=8)
        self.cell(0, self.margin, f"{self.user_name}'s ChatAdv Advising Session", 0, 0, "L")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", size=8)
        self.cell(0, self.margin, f"Generated on {self.date}", 0, 0, "L")
        self.cell(0, self.margin, f"Page {self.page_no()}", 0, 0, "R")

def draw_divider(pdf):
    y_position = pdf.get_y() + 3
    pdf.set_draw_color(200, 16, 45)
    pdf.set_line_width(1)
    pdf.line(pdf.margin, y_position, pdf.w - pdf.margin, y_position)
    pdf.ln(6)

def draw_heading(pdf, text):
    pdf.set_fill_color(255, 255, 255)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(200, 16, 46)
    pdf.multi_cell(0, pdf.margin, text, 0, "L", True)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)

def create_pdf(chat_messages, user_name):
    pdf = PDF(user_name, format="Letter")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=pdf.margin)

    # Document Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, pdf.margin, f"{user_name}'s Advising Session with ChatAdv on {pdf.date}", 0, 1, "C")
    pdf.ln(3)

    # Introductory Text
    draw_heading(pdf, "ChatAdv's Purpose")
    pdf.multi_cell(
        0, pdf.margin,
        "This document includes an export of the advising conversation with ChatAdv, "
        "an AI-powered chatbot designed to assist Farmer School of Business (FSB) students "
        "with their advising questions. While ChatAdv provides helpful information, "
        "students are encouraged to verify important details with their academic advisors.",
        0, "L", True
    )
    draw_divider(pdf)

    # ChatAdv's Interaction with the user
    draw_heading(pdf, f"{pdf.user_name}'s Interaction with ChatAdv")

    for message in chat_messages:
        role = message["role"]
        content = clean_text(re.sub(r"\n\s*\n", "\n", message["content"]))

        if role == "user":
            pdf.set_fill_color(255, 235, 224)
            pdf.multi_cell(0, pdf.margin, f"{pdf.user_name}:", fill=True)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.multi_cell(0, pdf.margin, "ChatAdv:", fill=True)

        for part in re.split(r"(```\w+?\n.*?```)", content, flags=re.DOTALL):
            if re.match(r"```(\w+)?\n(.*?)```", part, flags=re.DOTALL):  # code chunk
                code = re.findall(r"```(\w+)?\n(.*?)```", part, flags=re.DOTALL)[0][1]
                pdf.set_font("Courier", size=10)
                pdf.set_fill_color(230, 230, 230)
                pdf.multi_cell(0, pdf.margin, code, fill=True)
                pdf.set_font("Arial", size=11)
                pdf.ln(5)
            else:  # no code chunk - text
                if role == "user":
                    pdf.set_fill_color(255, 235, 224)
                    pdf.multi_cell(0, pdf.margin, part, fill=True)
                    pdf.ln(3)
                else:
                    pdf.set_fill_color(255, 255, 255)
                    pdf.multi_cell(0, pdf.margin, part, fill=True)
                    pdf.ln(6)

    # Disclaimer
    draw_heading(pdf, "Disclaimer")
    pdf.multi_cell(
        0, pdf.margin,
        "This document is a record of an AI-assisted advising session and should not be considered "
        "as official academic advice. Students are encouraged to verify all information and "
        "discuss their academic plans with their assigned academic advisors.",
        0, "L", True
    )

    # Save the pdf and return the file path
    pdf_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(pdf_output_path)
    return pdf_output_path
