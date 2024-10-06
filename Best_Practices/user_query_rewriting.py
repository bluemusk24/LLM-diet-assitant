import lancedb
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize necessary models and tools
nltk.download('wordnet')
spell = SpellChecker()

db_path = '/home/bluemusk/diet-assistant/lancedb'
db = lancedb.connect(db_path)
table_1 = db.open_table('diet_table')
table_2 = db.open_table('new_diet_table')

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Query rewriting functions
def spelling_correction(query):
    words = query.split()
    corrected_words = [spell.correction(word) if spell.unknown([word]) else word for word in words]
    return " ".join(corrected_words)

def expand_query_with_synonyms(query):
    words = query.split()
    expanded_query = []

    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            expanded_query.append(f"({word} OR {' OR '.join(synonyms)})")
        else:
            expanded_query.append(word)

    return " ".join(expanded_query)

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())

    return list(set(synonyms))[:3]


def search_text(query):
    text_search = table_1.search(query, query_type="fts").limit(5).select(["text"]).to_list()
    return text_search

def search_vector(query):
    query_embedding = embed_model.encode(query).tolist()
    semantic_search = table_2.search(query_embedding, query_type='vector', vector_column_name='embedding').limit(5).select(['text']).to_list()
    return semantic_search


def rank_documents(query, weight_text=0.5, weight_vector=0.5):
    full_text_results = search_text(query)
    semantic_results = search_vector(query)

    combined_results = []

    for result in full_text_results:
        result['combined_score'] = weight_text
        combined_results.append(result)

    for result in semantic_results:
        result['combined_score'] = weight_vector
        combined_results.append(result)

    ranked_results = sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)

    return ranked_results


def run_rag_pipeline(query):
    corrected_query = spelling_correction(query)

    expanded_query = expand_query_with_synonyms(corrected_query)

    print("Rewritten Query:", expanded_query)

    ranked_documents = rank_documents(expanded_query)

    top_documents = [doc['text'] for doc in ranked_documents[:5]]

    print("\nRanked Documents:")
    for idx, document in enumerate(top_documents, 1):
        print(f"Document {idx}: {document}")

if __name__ == "__main__":
    query = input("Enter your search query: ")
    run_rag_pipeline(query)
