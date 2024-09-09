Problem statement:

For many researchers, finding the right papers to reference can be challenging and time-consuming. This process often hinders productivity and diverts the researcher's attention away from their primary work.

The Solution:

The solution is an RAG system that uses vector search and the Gemma2-9B model to retrieve and present the most relevant paper for the user's purposes.

How it works:
- User writes a query
- The query is made into a vector using the multi-qa-MiniLM-L6-cos-v1 model from the SentenceTransformers library
- Use cosine similarity vector search to search to through a databse using ElasticSearch
- Pass the resulting documents back to the LLM to formulate a response

How to run the code:

1. Run `pip install -r project/requirements.txt` to install all required dependencies
2. Index the data in evaluation.ipynb
3. Run `streamlit run project/main.py`

Evalutions(for code check evaluation.ipynb):

Through previous testing, I found that the Gemma models have far better and predictable behavior compared to the llama models(even the 70b model). 

Initial prompt for :




Dataset used:
https://huggingface.co/datasets/rubrix/research_papers_multi-label 
    – Note: Only 1000 out of the 21000 rows were actually indexed due to hardware limitations on my local computer