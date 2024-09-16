from elasticsearch import Elasticsearch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

es_client = Elasticsearch('http://elasticsearch:9200')
ds = load_dataset("rubrix/research_papers_multi-label", split='train', streaming=True)
ds_head = ds.take(1000)
papers=list(ds_head)
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

print('Loaded Dataset...')
print('Starting Tokenization...')

for paper in tqdm(papers):
  title = paper['inputs']['title']
  abstract = paper['inputs']['abstract']
  paper['v_t'] = model.encode(title)
  paper['v_a'] = model.encode(abstract)
  paper['v_at'] = model.encode(title + ' ' + abstract)

index_settings = {
    'settings':
    {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    'mappings':
    {
        'properties':
        {
            'inputs' : {
                'properties': {
                    'abstract' : {'type': 'text'},
                    'title' : {'type': 'text'},
                }
            },
            'v_t': {
                "type": "dense_vector", 
                "dims": 384, 
                "index": True, 
                "similarity": "cosine"
            },
            'v_a': {
                "type": "dense_vector", 
                "dims": 384, 
                "index": True, 
                "similarity": "cosine"
            },
            'v_at': {
                "type": "dense_vector", 
                "dims": 384, 
                "index": True, 
                "similarity": "cosine"
            },
            'pid': {'type': 'text'}
        }
    }
}

index_name = 'academic_papers'

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

print('Index created...')

for paper in tqdm(papers):
    es_client.index(index=index_name, document=paper)

print('Indexing Finished')