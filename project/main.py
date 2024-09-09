import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


client = OpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key='<insert API key here>'
)

es_client = Elasticsearch('http://localhost:9200')

def main():
    st.title('Academic Paper RAG')

    user_in = st.text_input('Enter your Query')

    if(st.button('Sumbit')):
        st.write(rag(user_in))


def prompt_builder(query, results):
    prompt_template =  """
    You are a librarian trying to help a researcher look for the research papers they are looking for.
    Make sure you only use the facts from the context to answer the question and nothing outside.
    
    Context: {context}

    Questions: {query}

    If you find something that works, respond in this format:

    Title: <insert the title here>

    Abstract: <insert Abstract here>
    """
    context = ''
    for doc in results:
        context += f"\nTitle: {doc['_source']['inputs']['title']}\nAbstract: {doc['_source']['inputs']['abstract']}"

    prompt = prompt_template.format(query=query, context = context)

    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model = 'llama3-70b-8192',
        messages = [{'role': 'user', 'content': prompt}]
    )

    return response.choices[0].message.content

def rag(query):
    result_docs = elastic_search(query)
    prompt = prompt_builder(query, result_docs)
    return llm(prompt)

def elastic_search(query, index='academic_papers_v'):
    v_q = model.encode(query)
    
    search_query = {
        "field": "v_at",
        "query_vector": v_q,
        "k": 5,
        "num_candidates":10000
    }

    es_results = es_client.search(
        index=index,
        knn=search_query, 
        source=['inputs', 'pid', 'v_at']
    )

    return es_results['hits']['hits']

if __name__ == '__main__':
    main()