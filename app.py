import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate 
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap = 10000)
    chunks = text_splitter.split_text(text)
    return chunks 


def get_vector_store(text_chunks ):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')


def get_conversational_chain():
    prompt_template = """Answer the question as deatild as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", do not provide wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}?\n
     
    
    Answer: 
    
    """

    model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt = PromptTemplate (template=prompt_template,input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type='stuff',prompt= prompt )
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index',embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {'input_documents':docs,'question':user_question},return_only_outputs=True
    )
    print(response)
    st.write("reply:",response['output_text'])


def main():
    #st.set_page_config("chat with multiple PDFs ")
    st.header("chat with multiple PDF using Gemini ",divider='rainbow')

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files")
        if st.button("Submit "):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ")



if __name__ == "main":
    main()