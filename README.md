Problem statement:

Finding the right papers to reference can be challenging and time-consuming for many researchers. This process often hinders productivity and diverts the researcher's attention away from their primary work.

The Solution:

The solution is a RAG system that uses vector search and the Llama3-70B model to retrieve and present the most relevant paper for the user's purposes.

How it works:
- User writes a query
- The query is made into a vector using multi-qa-MiniLM-L6-cos-v1 model from the SentenceTransformers library
- Use cosine similarity vector search to search to through a databse using ElasticSearch
- Pass the resulting documents back to the LLM to formulate a response

How to run the code:

1. Run `pip install -r project/requirements.txt` to install all required dependencies
2. Run `docker run -it \    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3`
4. Index the data by running index.py
5. Run `streamlit run project/main.py`
6. Visit localhost:8501 to see the project

OR

Use docker-compose to start the application in one go

Evalutions(for code check evaluation.ipynb):

- I used MRR and Hit Rate evaluations to check the retrieval accuracy.

For Text search:
  HR: 0.9803843074459567
  MRR: 0.9623932479316789

For Vector Search(against the similarity of the vector of the abstract):
  HR: 0.9597678142514011
  MRR: 0.9245563117160396

For VS(against vector of the title):
  HR: 0.8060448358686949
  MRR: 0.7277822257806249

For VS(against vector of abstract+title):
  HR: 0.9621697357886309
  MRR: 0.9252168401387777 

Through previous testing, I found that the Gemma models have far better and more predictable behavior compared to the llama models(even the 70b model), but the llama models tend to be more user-friendly and suited for the final LLM response.

Dataset used:
https://huggingface.co/datasets/rubrix/research_papers_multi-label 
    – Note: Only 1000 out of the 21000 rows were actually indexed due to hardware limitations on my local computer
