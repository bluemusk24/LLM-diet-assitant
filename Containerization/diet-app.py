import streamlit as st
import time

import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Extract data from the PDF document
def load_pdf(data):
    loader = PyPDFLoader(data)
    documents = loader.load()
    return documents

extracted_data = load_pdf('Introduction to Nutrition Science, LibreTexts Project.pdf')


# Split the extracted pdf documents into chunks using LangChain's text splitter
def split_pdf(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

chunks = split_pdf(extracted_data)


# Generate embeddings using SentenceTransformer
def generate_embeddings(chunks, model_name: str = "all-MiniLM-L6-v2"):
    embed_model = SentenceTransformer(model_name)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_model.encode(texts)
    return embeddings

embeddings = generate_embeddings(chunks)


# Load chunks into created table in Lancedb
def load_chunks_into_lancedb(chunks, embeddings, db_path: str, table_name: str):
    
    db = lancedb.connect(db_path)
    
    custom_schema = pa.schema([
        pa.field('chunk_id', pa.int32()),
        pa.field('text', pa.string()),
        pa.field('embedding', pa.list_(pa.float64()))
    ])

    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=custom_schema)
    else:
        table = db.open_table(table_name)
    
    data = {
        "chunk_id": [],
        "text": [],
        "embedding": []
    }

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        data["chunk_id"].append(i)
        data["text"].append(chunk.page_content)
        data["embedding"].append(embedding.tolist()) 

    df = pd.DataFrame(data)

    print(db.table_names())
    print(f"Inserted {len(chunks)} chunks with embeddings into LanceDB table '{table_name}'.")

    return df, table


db_path = "/home/bluemusk/diet-assistant/lancedb"   
table_name = "diet_table"
df, table = load_chunks_into_lancedb(chunks, embeddings, db_path, table_name)


def add_dataframe_to_table(dataframe):
    table.add(dataframe)
    print(f"Added {len(chunks)} chunks and {len(embeddings)} embeddings to the table.")
    return table

table = add_dataframe_to_table(df)


# Convert embeddings list to FixedSizeListArray
def convert_to_fixed_size_list(embeddings, embedding_size):
    flattened_embeddings = []
    
    for embedding in embeddings:
        if len(embedding) != embedding_size:
            raise ValueError(f"Embedding size {len(embedding)} does not match the expected size {embedding_size}")
        flattened_embeddings.extend(embedding)
    
    numpy_array = np.array(flattened_embeddings, dtype=np.float32)
    
    fixed_size_list = pa.FixedSizeListArray.from_arrays(
        pa.array(numpy_array),
        list_size=embedding_size
    )
    
    return fixed_size_list

arrow_table = table.to_arrow()
embedding_size = 384
fixed_size_embeddings = convert_to_fixed_size_list(arrow_table.to_pandas()['embedding'], embedding_size)

updated_table = arrow_table.set_column(
    arrow_table.schema.get_field_index('embedding'),    
    'embedding',                                           
    fixed_size_embeddings                                  
)


db = lancedb.connect(db_path)

if 'new_diet_table' not in db.table_names():
    db.create_table('new_diet_table', updated_table)
    print("Created 'new_diet_table'.")
else:
    print("'new_diet_table' already exists.")

new_table = db.open_table('new_diet_table')


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


def search_text(query, table, limit=5):
    table.create_fts_index("text")
    text_search = table.search(query, query_type="fts").limit(limit).select(["text"]).to_list()
    return text_search


def search_vector(query, table, limit=5):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embed_model.encode(query).tolist()
    table.create_index(metric='cosine', vector_column_name='embedding', index_type='IVF_PQ')
    semantic_search = table.search(query_embedding, query_type='vector', vector_column_name='embedding').limit(limit).select(['text']).to_list()
    return semantic_search


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


def llm(prompt, model, tokenizer, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_pipeline(query, model, tokenizer, search_type, limit=5, max_tokens=100):
    if search_type == 'semantic':
        search_results = search_vector(query, new_table, limit=limit)
    else:
        search_results = search_text(query, table, limit=limit)

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
