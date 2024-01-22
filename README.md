# Streamlit Chatbot on PDF Project

## Overview

This project leverages the capabilities of Streamlit to create a chatbot that interacts with PDF documents. The workflow involves using the PdfReader package to extract text from PDF files, chunking the text for efficient processing, and employing OpenAIEmbeddings and FAISS search to create a vector store. The vector store is then utilized with the ChatOpenAI language model to build a sophisticated chatbot capable of understanding and responding to user queries using langchain.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AhmedElsaidy1995/chat_pdf.git
   cd chat_pdf
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up OpenAI API key:

   Obtain your OpenAI API key from [https://platform.openai.com/signup](https://platform.openai.com/signup) and set it as an environment variable

## Usage

1. Start the app `streamlit run app.py`

2. Load the Pdf you would like to ask questions

3. Ask questions and get the answers

## Screenshot
![2024-01-22 6-54-55 PM](https://github.com/AhmedElsaidy1995/chat_pdf/assets/45700852/95edb42a-c669-4531-be01-fec4898a19c7)

   



