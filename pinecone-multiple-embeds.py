import logging
import time

import tiktoken
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import os
import uuid
import openai
import pinecone
from tqdm import tqdm
import streamlit as st
from pathlib import Path
from os import listdir
from os.path import isfile, join
from transformers import AutoModel, AutoTokenizer
from transformers.utils.import_utils import is_torch_npu_available

from huggingface_hub import login

hf_token = "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"

# Ensure you're logged in to Hugging Face
login(token=hf_token)
logging.basicConfig(level=logging.DEBUG)

# Access environment variables
PINECONE_API_KEY = "24d65218-04fb-4688-99b3-871d994833bb"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAMES = {
    "GTE Base": "rnd-kb-gte-base",
    "BGE Base": "rnd-kb-bge-base",
    "E5 Base": "rnd-kb-e5-base"
}

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)


# Define the models
def load_model(model_name):
    if model_name == "GTE Base":
        model_path = 'Alibaba-NLP/gte-base-en-v1.5'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    elif model_name == "BGE Base":
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return model, None
    elif model_name == "E5 Base":
        model = SentenceTransformer('intfloat/e5-base-v2')
        return model, None
    return None, None


# Define encoding functions
def encode_texts_gte_base(model, tokenizer, texts):
    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

    # Get the model outputs
    outputs = model(**batch_dict)

    # Extract embeddings from the [CLS] token (first token of the output)
    embeddings = outputs.last_hidden_state[:, 0]

    # Normalize the embeddings (optional but recommended)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.tolist()


def encode_texts_bge(model, texts):
    return model.encode(texts).tolist()


def encode_texts_e5(model, texts):
    return model.encode(["query: " + text for text in texts]).tolist()


def count_tokens(text):
    # Tokenize the text
    tokens = encoding.encode(text)
    return len(tokens)


def count_bytes(text):
    return len(text.encode('utf-8'))


def read_text(file_path, encodings=['utf-8', 'latin-1', 'cp1252']):
    with open(file_path, 'rb') as file:
        raw_bytes = file.read()

    for encoding in encodings:
        try:
            raw_text = raw_bytes.decode(encoding)
            return raw_text
        except UnicodeDecodeError:
            pass
    return None


def process_data(content_type, k_ownership_source, model_name):
    data = []
    new_data = []
    exceeded_file = []
    exceeded_tokens = []

    directory_path = r"C:/Users/USER/Documents/Knowledge-Base-Books/Text Files - U.S. Master-Employee Benefits Guide 2020"
    starting_index = get_vector_count(model_name)

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
        if byte_count < 40000:
            new_data.append({
                'id': str(new_i),
                'text': data[i],
                'content_type': content_type,
                'k_ownership_source': k_ownership_source,
            })
        elif byte_count >= 40000:
            exceeded_tokens.append(token_count)
            exceeded_file.append(data[i])

    print("Processed Data")
    upsert_to_pinecone(new_data, model_name)
    return new_data


def upsert_to_pinecone(new_data, model_name):
    start_time = time.time()
    model, tokenizer = load_model(model_name)
    load_time = time.time() - start_time
    logging.debug(f"Model loaded in {load_time} seconds")
    batch_size = 2  # Reduce the batch size to manage memory usage better
    index = pc.Index(PINECONE_INDEX_NAMES[model_name])

    for i in tqdm(range(0, len(new_data), batch_size)):
        print("In loop")
        meta_batch = new_data[i:i + batch_size]
        ids_batch = [x['id'] for x in meta_batch]
        texts = [x['text'] for x in meta_batch]

        if model_name == "GTE Base":
            res = encode_texts_gte_base(model, tokenizer, texts)
        elif model_name == "BGE Base":
            res = encode_texts_bge(model, texts)
        elif model_name == "E5 Base":
            res = encode_texts_e5(model, texts)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        meta_batch = [{
            'text': x['text'],
            'content_type': x['content_type'],
            'k_ownership_source': x['k_ownership_source']
        } for x in meta_batch]

        to_upsert = list(zip(ids_batch, res, meta_batch))
        print("to upsert: ", to_upsert)
        index.upsert(vectors=to_upsert)


def retrieve_context(query, model_name):
    model, tokenizer = load_model(model_name)
    index = pc.Index(PINECONE_INDEX_NAMES[model_name])

    if model_name == "GTE Base":
        query_embedding = encode_texts_gte_base(model, tokenizer, [query])[0]
    elif model_name == "BGE Base":
        query_embedding = encode_texts_bge(model, [query])[0]
    elif model_name == "E5 Base":
        query_embedding = encode_texts_e5(model, [query])[0]

    res = index.query(query_embedding, top_k=3, include_metadata=True)
    contexts = [x['metadata']['text'] for x in res['matches']]
    return contexts


def get_vector_count(model_name):
    index = pc.Index(PINECONE_INDEX_NAMES[model_name])
    index_stats = index.describe_index_stats()
    vector_count = index_stats["total_vector_count"]
    return vector_count


def delete_database(model_name, metadata):
    index = pc.Index(PINECONE_INDEX_NAMES[model_name])
    index.delete(
        filter={
            "k_ownership_source": {"$eq": metadata}
        }
    )


# Streamlit app
def main():
    st.title("PromptEnforce - Multi-Model Pinecone Retrieval")

    query_option = st.radio("Select Query Option:", ["Upload Documents", "Delete Database", "Get Vectors"])
    model_option = st.selectbox("Select Embedding Model:", ["GTE Base", "BGE Base", "E5 Base"])

    if query_option == "Upload Documents":
        selected_directory = st.text_input("Enter or paste a directory path")
        if os.path.isdir(selected_directory):
            st.write(f"Selected Directory: {selected_directory}")
        content_type = st.text_input("Enter the content_type: ")
        k_ownership_source = st.text_input("Enter the k_ownership_source: ")

        upsert_button = st.button("Upsert Vectors")
        if upsert_button:
            st.subheader("Processed Data:")
            new_data = process_data(content_type, k_ownership_source, model_option)
            st.json(new_data)
            st.write("Vectors upserted")

    elif query_option == "Delete Database":
        vec_num = get_vector_count(model_option)
        select_metadata = st.text_input("Enter metadata of database: ")
        if vec_num != 0:
            st.write("Are you sure you want to delete this collection of data?")
            confirmation = st.button("Yes")
            if confirmation:
                delete_database(model_option, select_metadata)
                st.write(f"Database has been cleared of all vectors of {select_metadata}")
        else:
            st.write("There are no vectors in the database")

    elif query_option == "Get Vectors":
        query = st.text_input("Write your query: ")

        get_vectors_button = st.button("Get Relevant Vectors")

        if get_vectors_button:
            st.subheader("Here are the contexts retrieved from Pinecone:")
            contexts = retrieve_context(query, model_option)
            st.json(contexts)


if __name__ == "__main__":
    main()

# streamlit run pinecone-multiple-embeds.py
