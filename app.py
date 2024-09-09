import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Set your API key
API_KEY = 'AIzaSyCrbeU4QGZYHXR2AIAfeiko5AN8NCerQ24'  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Helper functions
def embed_content(title, text, model='models/embedding-001', task_type='retrieval_document'):
    response = genai.embed_content(model=model, content=text, task_type=task_type, title=title)
    return response["embedding"]

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_top_chunks(query, dataframe, top_n=3, model='models/embedding-001'):
    query_response = genai.embed_content(model=model, content=query, task_type='retrieval_query')
    query_embedding = query_response["embedding"]

    document_embeddings = np.stack(dataframe['Embeddings'])
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top_n indices

    return dataframe.iloc[top_indices]['Text'].tolist()

def make_prompt(query, relevant_passages):
    passages = " ".join(relevant_passages)
    escaped = passages.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(f"""\
    You are a helpful and informative bot that answers questions using text from the reference passages included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGES: '{escaped}'

    ANSWER:
    """)
    return prompt

# Streamlit UI
st.title("PDF Question Answering App")

# PDF Upload
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf is not None:
    # Extract text from uploaded PDF
    document_text = extract_text_from_pdf(uploaded_pdf)

    # Create chunks
    chunks = create_chunks(document_text)
    
    # Embed the chunks
    df = pd.DataFrame(chunks, columns=['Text'])
    df['Title'] = ['Chunk ' + str(i+1) for i in range(len(chunks))]
    df['Embeddings'] = df.apply(lambda row: embed_content(row['Title'], row['Text']), axis=1)

    # User Query
    query = st.text_input("Ask a question")

    if query:
        # Find relevant passages
        top_passages = find_top_chunks(query, df, top_n=3)
        
        # Create a prompt for the generative model
        prompt = make_prompt(query, top_passages)

        # Get the answer from the generative model
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        answer = model.generate_content(prompt)

        # Display the answer
        st.write("Answer:", answer.text)
