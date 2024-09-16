# Problem Statement

Finding the right papers to reference can be challenging and time-consuming for many researchers. This process often hinders productivity and diverts the researcher's attention away from their primary work.

# The Solution

The solution is a RAG system that uses vector search and the Llama3-70B model to retrieve and present the most relevant paper for the user's purposes.

## How It Works

1. User writes a query.
2. The query is converted into a vector using the `multi-qa-MiniLM-L6-cos-v1` model from the SentenceTransformers library.
3. Use cosine similarity vector search to search through a database using ElasticSearch.
4. Pass the resulting documents back to the LLM to formulate a response.

## How to Run the Code

1. Run `pip install -r project/requirements.txt` to install all required dependencies.
2. Run the following Docker command to start ElasticSearch:
    ```sh
    docker run -it \
        --rm \
        --name elasticsearch \
        -p 9200:9200 \
        -p 9300:9300 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        docker.elastic.co/elasticsearch/elasticsearch:8.4.3
    ```
3. Index the data by running `index.py`.
4. Run `streamlit run project/main.py`.
5. Visit `localhost:8501` to see the project.

### OR

Use `docker-compose` to start the application in one go, then navigate to [http://localhost:8501](http://localhost:8501) to use the application.
  - If you get a timeout from HuggingFace, restart the Streamlit application.
  - Before you use this, make sure to modify the `docker-compose.yaml` file accordingly.

## Evaluations (for code check `evaluation.ipynb`)

### Search Evaluations

I used MRR and Hit Rate evaluations to check the retrieval accuracy.

- For Text search: 
  - HR: 0.9803843074459567 
  - MRR: 0.9623932479316789

- For Vector Search (against the similarity of the vector of the abstract): 
  - HR: 0.9597678142514011 
  - MRR: 0.9245563117160396

- For VS (against vector of the title): 
  - HR: 0.8060448358686949 
  - MRR: 0.7277822257806249

- For VS (against vector of abstract + title): 
  - HR: 0.9621697357886309 
  - MRR: 0.9252168401387777

Through previous testing, I found that the Gemma models have far better and more predictable behavior compared to the llama models (even the 70B model), but the llama models tend to be more user-friendly and suited for the final LLM response.

## Dataset Used

[Research Papers Multi-Label Dataset](https://huggingface.co/datasets/rubrix/research_papers_multi-label) – Note: Only 1000 out of the 21000 rows were actually indexed due to hardware limitations on my local computer.

## References

[Data-talks LLM-Zoomcamp Github](https://github.com/DataTalksClub/llm-zoomcamp.git) - Utlized methodology that was discussed in this course
