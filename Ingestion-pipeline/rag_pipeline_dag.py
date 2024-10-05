from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Connect to Lancedb and open tables
db_path =  '/home/bluemusk/diet-assistant/lancedb'
db = lancedb.connect(db_path)
table_1 = db.open_table('diet_table')
table_2 = db.open_table('new_diet_table')

# Define RAG pipeline steps
def search_text(query):
    text_search = table_1.search(query, query_type="fts").limit(5).select(["text"]).to_list()
    return text_search

def search_vector(query):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embed_model.encode(query).tolist()
    semantic_search = table_2.search(query_embedding, query_type='vector', vector_column_name='embedding').limit(5).select(['text']).to_list()
    return semantic_search

def build_prompt(query, search_results):
    prompt_template = f"""
    You are a diet assistant. Based on the provided context, answer the following question:

    QUESTION: {query}
    CONTEXT: {search_results}
    """.strip()
    return prompt_template

def generate_response(prompt):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the RAG pipeline task
def rag_pipeline(**context):
    query = context['dag_run'].conf.get('query', 'What is a balanced diet?')
    search_type = context['dag_run'].conf.get('search_type', 'full-text')
    
    if search_type == 'semantic':
        search_results = search_vector(query)
    else:
        search_results = search_text(query)
    
    prompt = build_prompt(query, search_results)
    response = generate_response(prompt)
    
    print("Generated Response:", response)

# Define DAG
default_args = {
    'owner': 'airflow',
    'start_date': pendulum.today('UTC').add(days=-1),
    'retries': 1,
}

with DAG(
    dag_id='rag_pipeline_dag',
    default_args=default_args,
    schedule='@daily', 
    catchup=False,
) as dag:

    rag_task = PythonOperator(
        task_id='run_rag_pipeline',
        python_callable=rag_pipeline,
    )

rag_task


