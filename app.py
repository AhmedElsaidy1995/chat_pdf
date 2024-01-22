# python -m streamlit run app.py
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def get_pdf_text(pdf_files):
    # Get the text from PDF
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    # Split the text to make search faster
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )
    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    # Create vectorstore to search the data
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store):
    # Create LLM Chain
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain


def main():
    # Load .env which should have your OPENAI API KEY
    load_dotenv()
    # Set page title
    st.set_page_config(page_title='Chat with Your PDF')
    # Create llm_chain 
    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = None
    # Create a variable to store chat history in
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'disabled' not in st.session_state:
        st.session_state.disabled = True

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("", type=['pdf'], accept_multiple_files=True)

        if st.button("Process the File"):
            with st.spinner("Processing your PDFs..."):

                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                
                # Create Vector Store
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain
                st.session_state.llm_chain =  get_conversation_chain(vector_store)
                st.session_state.disabled = False

                st.write("Processing Done")
                st.write("Now you can chat with your PDF")

    # Display the messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Show the prompt 
    if prompt := st.chat_input("Ask me about Your PDF?",disabled=st.session_state.disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # get the answer of the question
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.llm_chain({'question':prompt})
                st.write(response['answer'])
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})



if __name__ == '__main__':
    main()