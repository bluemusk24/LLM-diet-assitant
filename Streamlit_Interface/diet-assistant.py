import streamlit as st
import time

import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Connect to Lancedb locally
db_path = '/home/bluemusk/diet-assistant/lancedb'
db = lancedb.connect(db_path)

# load the tables(text search and semantic search) from Lancedb
table_1 = db.open_table('diet_table')
table_2 = db.open_table('new_diet_table')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Full-text search retrieval function
def search_text(query, table, limit=5):
    text_search = table.search(query, query_type="fts").limit(limit).select(["text"]).to_list()
    return text_search

# Semantic search retrieval function
def search_vector(query, table, limit=5):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embed_model.encode(query).tolist()
    semantic_search = table.search(query_embedding, query_type='vector', vector_column_name='embedding').limit(limit).select(['text']).to_list()
    return semantic_search

# Build a prompt for the model
def build_prompt(query, search_results, tokenizer, max_length=512):
    prompt_template = f"""
    You are a diet assistant. You are performing either full-text and semantic search. 
    Based on the provided context, answer the following question completely and coherently.
    Use the information from the CONTEXT to provide a detailed and full response to the QUESTION.
    Ensure your response is comprehensive and complete, avoiding any abrupt or partial endings.

    QUESTION: {{question}}
    CONTEXT: {{context}}
    """.strip()

    context = ""
    for item in search_results:
        context += f'{item.get("text", "")}\n{item.get("embedding", "")}\n\n'

    prompt = prompt_template.format(question=query, context=context).strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    truncated_prompt = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    
    return truncated_prompt

# Generate a response from the model
def llm(prompt, model, tokenizer, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Full RAG pipeline
def rag_pipeline(query, model, tokenizer, search_type, limit=5, max_tokens=100):
    if search_type == 'semantic':
        search_results = search_vector(query, table_2, limit=limit)
    else:
        search_results = search_text(query, table_1, limit=limit)

    prompt = build_prompt(query, search_results, tokenizer)
    answer = llm(prompt, model, tokenizer, max_tokens=max_tokens)
    return answer

# Streamlit app interface
def main():
    st.title("RAG Diet-Assistant")

    query = st.text_input("Enter your query:")

    search_type = st.radio("Choose search type:", ('Full-text Search', 'Semantic Search'))

    result_limit = st.slider("Number of results:", 1, 20, 5) 
    token_limit = st.slider("Max token output:", 50, 300, 100)

    if st.button("Ask"):
        if query.strip():
            with st.spinner('Processing...'):
                if search_type == 'Full-text Search':
                    search_method = 'full-text'
                else:
                    search_method = 'semantic'
                output = rag_pipeline(query, model, tokenizer, search_type=search_method, limit=result_limit, max_tokens=token_limit)
                st.success("Completed!")
                st.write(output)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
