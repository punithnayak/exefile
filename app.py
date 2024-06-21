import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity



# Introduce a session state variable to track processing completion
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'uploaded_excle_files' not in st.session_state:
    st.session_state.uploaded_excle_files = None

if 'uploaded_excle_files_on' not in st.session_state:
    st.session_state.uploaded_excle_files_on = False

def get_pdf_text(pdf_docs):
    data = []
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path)
        data.extend(loader.load())
        os.remove(tmp_file_path)

    return data

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(text)
    return all_splits


def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_documents(text_chunks, embeddings)

def handle_userinput(user_question):
    if st.session_state.processing_complete:
        if st.session_state.vectorstore is not None:
            docs = st.session_state.vectorstore.similarity_search(user_question)
            # finaldata = docs[2].page_content if len(docs) > 2 else ""
            print('***********************************')
            print(docs)
            print('***********************************')
            # finalresponse = getLLamaresponse(user_question, 100, finaldata)
            st.write(docs[0].page_content)
            st.write(docs[1].page_content)
            st.write(docs[2].page_content)
            st.write(docs[3].page_content)
        else:
            st.write("Vectorstore is not initialized")
    else:
        st.write("Processing is still ongoing.")

def getLLamaresponse(input_text, no_words, finaldata):
    llm = CTransformers(model='.\models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 6000,
                                'temperature': 0.01})
    template = """
    Using these words {finaldata}
        Give clear and simple instructions in points for a common person on the topic '{input_text}'.
        The instructions should be within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["input_text", 'no_words', 'finaldata'], template=template)
    response = llm(prompt.format(input_text=input_text, no_words=no_words, finaldata=finaldata))
    return response

def find_most_similar_question(user_question, questions, answers):
    # Combine the user question with the existing questions
    all_questions = questions + [user_question]
    
    # Step 3: Compute the similarity between the user's question and the questions in the Excel file
    vectorizer = TfidfVectorizer().fit_transform(all_questions)
    vectors = vectorizer.toarray()
    
    # Reshape the user's question vector to 2D
    user_question_vector = vectors[-1].reshape(1, -1)
    question_vectors = vectors[:-1]
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(user_question_vector, question_vectors)
    
    # Find the index of the most similar question
    most_similar_index = cosine_similarities.argmax()
    
    # Step 4: Return the answer corresponding to the most similar question
    return answers[most_similar_index]

def xlsx_search(user_question):
    answers = []
    questions = []

    for file in st.session_state.uploaded_excle_files:
        df = pd.read_excel(file)
        df.columns = ['Question', 'Answer']
        df.dropna(subset=['Question', 'Answer'], inplace=True)
        questions.extend(df['Question'].tolist())
        answers.extend(df['Answer'].tolist())
        
    final_answer = find_most_similar_question(user_question, questions, answers)
    st.write(final_answer)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.header("Chat with multiple PDFs and Excel:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.uploaded_excle_files_on:
            xlsx_search(user_question)
        else:
            handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                st.session_state.processing_complete = False  # Reset processing flag
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.session_state.processing_complete = True  # Update processing flag after completion
        st.session_state.uploaded_excle_files_on = st.toggle("Activate Excle Serach",value=False)
        if st.session_state.uploaded_excle_files_on:
            st.session_state.uploaded_excle_files = st.file_uploader(
                "Upload your Excle files here ", accept_multiple_files=True)

if __name__ == '__main__':
    main()
