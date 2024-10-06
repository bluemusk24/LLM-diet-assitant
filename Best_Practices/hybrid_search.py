import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Connect to LanceDB and open tables
db_path = '/home/bluemusk/diet-assistant/lancedb'
db = lancedb.connect(db_path)
table_1 = db.open_table('diet_table')
table_2 = db.open_table('new_diet_table')


def search_text(query):
    text_search = table_1.search(query, query_type="fts").limit(5).select(["text"]).to_list()
    return text_search


def search_vector(query):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embed_model.encode(query).tolist()
    semantic_search = table_2.search(query_embedding, query_type='vector', vector_column_name='embedding').limit(5).select(['text']).to_list()
    return semantic_search


def build_prompt(query, search_results):
    combined_results = " ".join([result['text'] for result in search_results])
    
    prompt_template = f"""
    You are a diet assistant. Based on the provided context, answer the following question:

    QUESTION: {query}
    CONTEXT: {combined_results}
    """.strip()
    return prompt_template


def generate_response(prompt):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_new_tokens=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def rag_pipeline(query, search_type='both'):
    if search_type == 'full-text':
        search_results = search_text(query)
    elif search_type == 'semantic':
        search_results = search_vector(query)
    else:  # Default to both if no specific type is provided
        full_text_results = search_text(query)
        semantic_results = search_vector(query)
        search_results = full_text_results + semantic_results

    if not search_results:
        print("No search results found.")
        return

    prompt = build_prompt(query, search_results)
    response = generate_response(prompt)
    print("Generated Response:", response)


if __name__ == "__main__":

    query = input("Please enter your query: ")
    search_type = input("Enter search type (full-text/semantic/both, default is both): ") or "both"

    rag_pipeline(query=query, search_type=search_type)
