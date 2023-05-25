from dotenv import load_dotenv
import streamlit as st
import os
import openai
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract text from PDF
    if pdf is not None:
        pdf_read = PdfReader(pdf)
        text = ""
        for page in pdf_read.pages:
            text += page.extract_text()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        api_key = os.getenv("OPENAI_API_KEY")

        # Check if api exists
        if api_key is None:
            st.error("Please set OPENAI_API_KEY in .env file")
            return
        # Check if the api key is valid by sending a simple 'test' prompt
        try:
            openai.api_key = api_key
            st.write("Testing OPENAI_API_KEY...")
            list = openai.Engine.list()
            
        except Exception as e:
            st.write(e)
            st.error("Invalid OPENAI_API_KEY")
            return
        
        if not list is None: 
            st.success("Valid OPENAI_API_KEY: "+ str(len(list.data)) + " engines found")

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        embeddings.openai_api_key = api_key
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Ask a question

        user_question = st.text_input("Ask a question about the PDF:")
        if user_question is not None and user_question != "":
            st.write("Searching for answer...")
            docs = knowledge_base.similarity_search(user_question)
            
            if len(docs) > 0:
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
            else:
                st.error("No answer found")
if __name__ == "__main__":
    main()
