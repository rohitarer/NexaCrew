from django.shortcuts import render
from django.http import JsonResponse
import PyPDF2
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
from django.views.decorators.csrf import csrf_exempt

# Set your API key
API_KEY = 'AIzaSyCrbeU4QGZYHXR2AIAfeiko5AN8NCerQ24'  # Replace with your actual API key
genai.configure(api_key=API_KEY)

def embed_content(title, text, model='models/text-embedding-004', task_type='retrieval_document'):
    response = genai.embed_content(model=model, content=text, task_type=task_type, title=title)
    return response["embedding"]

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_top_chunks(query, dataframe, top_n=3, model='models/text-embedding-004'):
    query_response = genai.embed_content(model=model, content=query, task_type='retrieval_query')
    query_embedding = query_response["embedding"]

    document_embeddings = np.stack(dataframe['Embeddings'])
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get top_n indices

    return dataframe.iloc[top_indices]['Text'].tolist()

def make_prompt(query, relevant_passages):
    passages = " ".join(relevant_passages)
    escaped = passages.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
    You are a helpful and informative bot that answers questions using text from the reference passages included below. 
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
    strike a friendly and conversational tone. 
    Every time user asks a question, you go through all the chunks 
    If it is out of passage context then respond that it is irrelevant. 
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGES: '{escaped}'

    ANSWER:
    """
    return prompt

def index(request):
    return render(request, 'index.html')

def upload_pdf(request):
    if request.method == 'POST' and request.FILES.get('pdf'):
        pdf_file = request.FILES['pdf']
        document_text = extract_text_from_pdf(pdf_file)

        # Create chunks and embed
        chunks = create_chunks(document_text)
        df = pd.DataFrame(chunks, columns=['Text'])
        df['Title'] = ['Chunk ' + str(i+1) for i in range(len(chunks))]
        df['Embeddings'] = df.apply(lambda row: embed_content(row['Title'], row['Text']), axis=1)

        # Save dataframe as a CSV file on the server instead of using session
        df.to_csv('dataframe.csv', index=False)

        return JsonResponse({'message': 'PDF processed and embeddings created successfully.'})
    return JsonResponse({'error': 'No PDF uploaded.'}, status=400)

def ask_question(request):
    if not os.path.exists('dataframe.csv'):
        return JsonResponse({'error': 'No data available. Please upload a sample first.'}, status=400)

    try:
        df = pd.read_csv('dataframe.csv')
        df['Embeddings'] = df['Embeddings'].apply(eval).apply(np.array)
    except Exception as e:
        return JsonResponse({'error': f'Error processing embeddings: {str(e)}'}, status=500)

    query = request.POST.get('question', '')

    try:
        top_passages = find_top_chunks(query, df, top_n=3)
        prompt = make_prompt(query, top_passages)

        model = genai.GenerativeModel('gemini-1.5-flash')
        temperature = 0.1
        answer = model.generate_content(prompt)
    except Exception as e:
        return JsonResponse({'error': f'Error generating content: {str(e)}'}, status=500)

    return JsonResponse({'answer': answer.text})

