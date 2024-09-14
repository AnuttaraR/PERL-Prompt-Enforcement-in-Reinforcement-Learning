import logging
import os
import time
from pinecone import Pinecone
import tiktoken
import torch.nn.functional as F
import json
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from huggingface_hub import login

hf_token = "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"

# Ensure you're logged in to Hugging Face
login(token=hf_token, add_to_git_credential=True)
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


def retrieve_context(query, model_name):
    model, tokenizer = load_model(model_name)
    index = pc.Index(PINECONE_INDEX_NAMES[model_name])

    if model_name == "GTE Base":
        query_embedding = encode_texts_gte_base(model, tokenizer, [query])[0]
    elif model_name == "BGE Base":
        query_embedding = encode_texts_bge(model, [query])[0]
    elif model_name == "E5 Base":
        query_embedding = encode_texts_e5(model, [query])[0]

    res = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    contexts = []
    truncated_texts = []
    for x in res['matches']:
        contexts.append(x['metadata']['text'])
        truncated_text = x['metadata']['text'][:100]  # Get the first 100 characters
        truncated_texts.append({
            'truncated_text': truncated_text,
            'metadata': x['metadata']
        })
    return contexts, truncated_texts


# List of questions for evaluation
questions_list = [
    "What is a deductible?",
    "What is an out-of-pocket (OOP) maximum?",
    "What is a premium?",
    "What is a copayment (copay)?",
    "What is ERISA (Employee Retirement Income Security Act)?",
    "What is COBRA (Consolidated Omnibus Budget Reconciliation Act)?",
    "What is the Affordable Care Act (ACA)?",
    "Explain what in-network and out-of-network providers are.",
    "What is HIPAA (Health Insurance Portability and Accountability Act)?",
    "What is a Health Savings Account (HSA)?",
    "What is a Flexible Spending Account (FSA)?",
    "What is coinsurance?",
    "How to claim my medical insurance?",
    "How to appeal a denied insurance claim?",
    "How to check the status of my insurance claim?",
    "How can I find out what my insurance covers?",
    "How can I get more information about what more I can claim?",
    "How to choose the right health insurance plan for me?",
    "How to add a dependent to my health insurance plan?",
    "How to switch from one insurance provider to another?",
    "How to change my coverage during open enrollment?",
    "How to lower my insurance premium?",
    "How to use my Health Savings Account (HSA) effectively?",
    "How to calculate my out-of-pocket expenses?",
    "When can I claim my insurance?",
    "When does my insurance coverage begin?",
    "When can I change my health insurance plan?",
    "When do I need to enroll in Medicare?",
    "Where do I find information about my plans from my employer?",
    "Where can I get help understanding my insurance policy?",
    "Where can I compare different insurance plans?",
    "Where do I find out if my doctor is in-network?",
    "Where can I find in-network providers?",
    "Where can I go for urgent care?",
    "Where can I find mental health services covered by my insurance?",
    "Who do I contact for questions about my insurance?",
    "Who decides if a medical procedure is covered by insurance?",
    "Who can help me file an insurance claim?",
    "Who is eligible for COBRA coverage?",
    "Why do I need health insurance?",
    "Why did my insurance premium increase?",
    "Why is my claim being denied?",
    "Why is preventive care important and is it covered?",
    "Which insurance plan is best for families?",
    "Which type of health insurance plan should I choose?",
    "Explain the difference between a Health Maintenance Organization (HMO) and a Preferred Provider Organization (PPO).",
    "Which is more beneficial: paying a higher premium for lower copayments, or a lower premium with higher copayments?",
    "Which offers more savings: a Health Savings Account (HSA) or a Flexible Spending Account (FSA)?"
]


# Metric functions
def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method1)


def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


def calculate_meteor(reference, hypothesis):
    reference_tokens = reference.split()  # Tokenize the reference
    hypothesis_tokens = hypothesis.split()  # Tokenize the hypothesis
    return meteor_score([reference_tokens], hypothesis_tokens)


def calculate_bertscore(reference, hypothesis):
    P, R, F1 = bert_score([hypothesis], [reference], lang="en")
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}


def evaluate_metrics(reference, contexts):
    evaluations = {}
    for idx, chunk in enumerate(contexts):
        evaluations[f"Chunk {idx + 1}"] = {
            "BLEU": calculate_bleu(reference, chunk),
            "ROUGE": calculate_rouge(reference, chunk),
            "METEOR": calculate_meteor(reference, chunk),
            "BERTScore": calculate_bertscore(reference, chunk),
        }
    return evaluations


# Streamlit app
def main():
    st.title("PromptEnforce - Multi-Model Pinecone Evaluation")

    model_option = st.selectbox("Select Embedding Model:", ["GTE Base", "BGE Base", "E5 Base"])

    if st.button("Run Evaluation"):
        st.subheader("Evaluation Results")
        results = {}

        for question in questions_list:
            st.write(f"Evaluating for question: {question}")
            contexts, truncated_texts = retrieve_context(question, model_option)  # Retrieve top 3 chunks from Pinecone
            evaluations = evaluate_metrics(question, contexts)

            results[question] = {
                'evaluations': evaluations,
                'truncated_texts': truncated_texts  # Add truncated texts to results
            }
            st.json(evaluations)

        output_file = f"evaluation_results_{model_option}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        st.write(f"Evaluations saved to {output_file}.")


if __name__ == "__main__":
    main()
