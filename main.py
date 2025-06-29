import os
import streamlit as st
import pickle
import time
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv() # take environment variables from .env. (we need OPENAI_API_KEY)

file_path = "faiss_index"

st.title("News Researcher")

# Create a sidebar to take url inputs from the user
st.sidebar.title("News Article URLs")

# Create a list to store the URLs
urls = []
# User will be able to supply 3 URLs, so we create 3 text input fields
for i in range(3):
    # Get the URLs from the user and append them to the list
    url = st.sidebar.text_input(f"URL {i+1}", value="")
    urls.append(url)


# Create a placeholder to display the progress of the search
progress_placeholder = st.empty()

# Create a placeholder to display the question and the answer after the search
main_placeholder = st.empty()

# Create the OpenAI object to interact with the OpenAI API
llm = OpenAI(temperature=0.9, max_tokens=500)

# Create a button to start the search
clicked_search_button = st.sidebar.button("Search in URLs")

if clicked_search_button:
    # If the button is clicked, we will start by getting the documents from the URLs
    # We will use the UnstructuredURLLoader to load the documents
    url_loader = UnstructuredURLLoader(
        urls=urls
    )

    progress_placeholder.text("Loading data from URLs...")

    # Load the text data from the URLs
    data = url_loader.load()

    # Create a text splitter to split the text data into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n", ".", ","],
        chunk_size=1000
    )

    progress_placeholder.text("Splitting data into smaller chunks...")

    # Split the text data into smaller chunks
    chunks = text_splitter.split_documents(data)

    # Create an OpenAIEmbeddings object to get the embeddings of the chunks
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    progress_placeholder.text("Getting embeddings of the chunks...")

    # Save the embeddings of the chunks to FAISS index
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )
    vector_store.save_local(file_path)

    query = main_placeholder.text_input("Question: ")

    if query:
        print("Question:", query)
        if os.path.exists(file_path):
            print("FAISS index exists. Loading the index...")
            # Load the FAISS index
            # TODO: Fix save_local and load_local to work with the embeddings
            vector_store = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

            # Create a chain to search for the answer
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm = llm,
                retriever = vector_store.as_retriever(),
            )

            # Search for the answer
            progress_placeholder.text("Searching for the answer...")

            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer:")
            st.subheader(result['answer'])

            sources = result.get("sources", "")

            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") # Split the sources by new line
                for source in sources_list:
                    st.write(source)

