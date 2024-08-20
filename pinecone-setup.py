import tiktoken
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import uuid
import openai
import pinecone
from tqdm import tqdm
import streamlit as st
from pathlib import Path
from os import listdir
from os.path import isfile, join

# Access environment variables
PINECONE_API_KEY = "24d65218-04fb-4688-99b3-871d994833bb"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "rnd-insurance-kb"

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def count_tokens(text):
    # Tokenize the text
    tokens = encoding.encode(text)

    # Count the tokens
    token_count = len(tokens)

    return token_count


def count_bytes(text):
    return len(text.encode('utf-8'))


def read_text(file_path, encodings=['utf-8', 'latin-1', 'cp1252']):
    with open(file_path, 'rb') as file:
        raw_bytes = file.read()

    for encoding in encodings:
        try:
            raw_text = raw_bytes.decode(encoding)
            print(f"Decoded using {encoding} encoding:")
            print(raw_text)
            return raw_text
        except UnicodeDecodeError:
            print(f"Failed to decode using {encoding} encoding.")

    print("Unable to decode the file with the provided encodings.")
    return None


def process_data(content_type, k_ownership_source):
    data = []
    new_data = []
    exceeded_file = []
    exceeded_tokens = []

    directory_path = r"C:/Users/USER/Documents/Knowledge-Base-Books/Text Files - The Handbook of Employee Benefits"
    starting_index = get_vector_count(index)

    only_files = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    all_lines = []
    for file_name in only_files:
        file_path = Path(directory_path) / file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            all_lines.append(file_content.splitlines())
            data.append(file_content)

    for i in tqdm(range(0, len(data))):
        new_i = i + starting_index
        token_count = count_tokens(data[i])
        byte_count = count_bytes(data[i])
        if byte_count < 40960:
            new_data.append({
                'id': str(new_i),
                'text': data[i],
                'content_type': content_type,
                'k_ownership_source': k_ownership_source,
            })
        elif byte_count >= 40960:
            exceeded_tokens.append(token_count)
            exceeded_file.append(data[i])

    print("Exceeded Tokens: ", exceeded_tokens, "\n\n")

    for file in exceeded_file:
        print(" ".join(file.split()[:200]), "\n\n\n")

    upsert_to_pinecone(new_data)

    return new_data  # Return both new_data and file_names


# Upsert data to Pinecone index
def upsert_to_pinecone(new_data):
    batch_size = 40  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(new_data), batch_size)):
        meta_batch = new_data[i:i + batch_size]
        ids_batch = [x['id'] for x in meta_batch]
        texts = [x['text'] for x in meta_batch]

        res = embed_model.encode(texts).tolist()

        meta_batch = [{
            'text': x['text'],
            'content_type': x['content_type'],
            'k_ownership_source': x['k_ownership_source']
        } for x in meta_batch]

        to_upsert = list(zip(ids_batch, res, meta_batch))
        print("to upsert: ", to_upsert)
        index.upsert(vectors=to_upsert)


# Retrieve context from Pinecone
# No metadata filter used for now
def retrieve_context(query):
    xq = embed_model.encode(query).tolist()

    res = index.query(vector=xq, top_k=3, include_metadata=True)

    contexts = [
        x['metadata'] for x in res['matches']
    ]

    return contexts


def get_vector_count(index):
    index_stats = index.describe_index_stats()
    vector_count = index_stats["total_vector_count"]
    return vector_count


def delete_database(index):
    index.delete(delete_all=True)


# Streamlit app
def main():
    st.title("PromptEnforce - Basic Pinecone Retrieval")

    query_option = st.radio("Select Query Option:",
                            ["Upload Documents", "Delete Database", "Get Vectors"])

    if query_option == "Upload Documents":

        selected_directory = st.text_input("Enter or paste a directory path")
        if os.path.isdir(selected_directory):
            st.write(f"Selected Directory: {selected_directory}")
        content_type = st.text_input("Enter the content_type: ")
        k_ownership_source = st.text_input("Enter the k_ownership_source: ")

        upsert_button = st.button("Upsert Vectors")
        if upsert_button:
            st.subheader("Processed Data:")

            new_data = process_data(content_type, k_ownership_source)
            st.json(new_data)

            st.write("Vectors upserted")

    elif query_option == "Delete Database":

        vec_num = get_vector_count(index)

        if vec_num != 0:
            st.write("Are you sure you want to delete the database?")
            confirmation = st.button("Yes")

            if confirmation:
                delete_database(index)
                st.write("Database has been cleared of all vectors")

        else:
            st.write("There are no vectors in the database")

    elif query_option == "Get Vectors":

        query = st.text_input("Write your query: ")

        get_vectors_button = st.button("Get Relevant Vectors")

        if get_vectors_button:
            st.subheader("Here are the contexts retrieved from pinecone:")

            contexts = retrieve_context(query)
            st.json(contexts)




if __name__ == "__main__":
    main()

# streamlit run pinecone-setup.py
