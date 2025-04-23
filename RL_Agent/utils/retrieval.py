import json
import logging
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Logging Configuration
logging.basicConfig(
    level=print
)

# Pinecone Configuration
PINECONE_API_KEY = "24d65218-04fb-4688-99b3-871d994833bb"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "rnd-kb-e5-base"

# Initialize Pinecone Connection
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone connection established.")
except Exception as e:
    print(f"Pinecone connection failed: {e}")

# Load the E5 Base model for embeddings
try:
    embedding_model = SentenceTransformer("intfloat/e5-base-v2")
    print("SentenceTransformer E5-Base model loaded successfully.")
except Exception as e:
    print(f"Failed to load SentenceTransformer model: {e}")


def encode_texts_e5(texts):
    """Encode text queries into embeddings using E5-Base."""
    try:
        embeddings = embedding_model.encode(["query: " + text for text in texts]).tolist()
        print(f"Encoded {len(texts)} text(s) into embeddings.")
        return embeddings
    except Exception as e:
        print(f"Error during text encoding: {e}")
        return []


def retrieve_context(query, top_k=3):
    """Retrieve the top-k most relevant context chunks from Pinecone."""
    print(f"Retrieving context for query: '{query[:10]}' with top_k={top_k}")

    try:
        query_embedding = encode_texts_e5([query])[0]
        res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        contexts = [x['metadata']['text'] for x in res['matches']]
        combined_context = "\n ".join(contexts)
        print("COMBINED CONTEXT: ", combined_context)

        print(f"Retrieved {len(contexts)} relevant context chunk(s) from Pinecone.")
        return combined_context
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return ""


def load_questions_from_json(filepath):
    """Load questions from a structured JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} question(s) from {filepath}")
        return data
    except Exception as e:
        print(f"Failed to load JSON file {filepath}: {e}")
        return []


if __name__ == "__main__":
    test_query = "What is a benefits broker?"
    retrieved_text = retrieve_context(test_query, top_k=3)
    print("Retrieved Context:\n", retrieved_text)
