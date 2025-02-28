import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'
offload_folder = "D:/huggingface_offload"

# Make sure the folder exists
os.makedirs(offload_folder, exist_ok=True)
import torch
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from bart_scorer import BARTScorer
import pinecone
import json

# Pinecone setup
PINECONE_API_KEY = "24d65218-04fb-4688-99b3-871d994833bb"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "rnd-kb-e5-base"

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-32K-Instruct-AWQ", trust_remote_code=True)
print("Loading the full-precision model...")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-32K-Instruct-AWQ", trust_remote_code=True)
print("Model loaded successfully!")

# Quantize the model using PyTorch's dynamic quantization for CPUs

model_dir = "D:/huggingface_offload/llama_32k_model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
print(f"Quantized model saved to {model_dir}")

device = torch.device("cpu")
model.to(device)

# Initialize BARTScorer
bart_scorer = BARTScorer(device='cpu')

# Define max tokens for generation (similar to max_tokens in OpenAI API)
MAX_TOKENS = None


# Load the E5 Base model for embeddings
def load_model(model_name):
    if model_name == "E5 Base":
        model = SentenceTransformer('intfloat/e5-base-v2')
        return model, None
    return None, None


# Encoding function for E5 Base
def encode_texts_e5(model, texts):
    return model.encode(["query: " + text for text in texts]).tolist()


# Pinecone retrieval for top 3 results
def retrieve_context(query, model_name):
    model, _ = load_model(model_name)
    index = pc.Index(PINECONE_INDEX_NAME)
    query_embedding = encode_texts_e5(model, [query])[0]

    # Query Pinecone for top 3 results
    res = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    contexts = []
    for x in res['matches']:
        contexts.append(x['metadata']['text'])  # Extract the chunk from metadata

    # Combine the text of all chunks
    combined_context = " ".join(contexts)
    return combined_context


# Function to generate text using the Llama model
def get_llm_response(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    outputs = model.generate(inputs["input_ids"], num_return_sequences=1, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


# Evaluation metrics
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


def calculate_bart_score(bart_scorer, reference, hypothesis):
    return bart_scorer.score(hypothesis, reference)


def evaluate_metrics(reference, hypothesis, bart_scorer):
    return {
        "BLEU": calculate_bleu(reference, hypothesis),
        "ROUGE": calculate_rouge(reference, hypothesis),
        "METEOR": calculate_meteor(reference, hypothesis),
        "BERTScore": calculate_bertscore(reference, hypothesis),
        "BARTScore": calculate_bart_score(bart_scorer, reference, hypothesis)
    }


# Prompt construction
def construct_prompt(question, context, prompt_level):
    if prompt_level == "short":
        return f"{question}\n\nRelevant context: {context}"
    elif prompt_level == "medium":
        return f"As a knowledgeable assistant, answer the following question: {question}\n\nBased on the context: {context}. Please be concise."
    elif prompt_level == "long":
        return f"You are an expert on this topic. Answer the following question in detail, providing examples where applicable. Consider the context: {context}. The question is: {question}. Ensure clarity in your explanation."


# Array of dictionaries with questions and ground truths
questions_data = [
    {"question": "What is a 'benefits broker'?",
     "ground_truth": "A benefits broker is pretty much like any other insurance broker. They can help employers detc1mine what types of benefits work best for them and what plan options are available. In addition, benefit brokers can negotiate with plan providers to try to secure the lowest cost possible. A major advantage of working with benefit brokers is that there's no cost to the employer since brokers are paid in commissions from the plan providers based on the business the broker brings in.",
     "question_type": "what"},
    {"question": "What is an IRA?",
     "ground_truth": "An Individual Retirement Account or IRA is an account set up at a financial institution that allows an individual to save for retirement with tax-free growth (for 'Roth' IRAs) or on a tax­deferred basis (for 'traditional' IRAs).",
     "question_type": "what"},
    {"question": "What benefits are at the top of the list for employees and prospective employees?",
     "ground_truth": "Job applicants want paid time off and health insurance above all, but retirement plans are a close second. Other 'most wanted' benefits include overtime pay, paid medical leave, dental and vision coverage, and disability and life insurance. Flexible work schedules and the ability to work remotely are also becoming an important perk for employees.\nThe newest trend in employer-provided benefits? Wellness programs - physical wellness, financial coaching and emotional wellness programs. In fact a majority of employers in the U.S. now offer mental wellness programs, such as stress management, and financial help such as retirement planning or credit counseling.",
     "question_type": "what"},
    {"question": "How should employers prepare for a benefit plan audit?",
     "ground_truth": "Employers who have been notified of a scheduled audit of their benefit plan should be sure to:\n• Review and familiarize themselves with their plan documents and prepare copies for the auditor.\n• Contact the auditor ahead of time to discuss timing, infom1ation requests and accommodations.\n• Be prepared to show a log of their internal audit controls related to employee enrollment procedures, bow employee contributions are withheld and remitted to the plan, how employer contributions are determined and remitted to the plan, how funds are controlled and how loans and distributions are processed.\n• Collect and review all agreements with service providers including financial institutions and third-party administrators.\n• Be prepared to produce copies of all financial records associated with the benefit plan.",
     "question_type": "how"},
    {"question": "How does a health care Flexible Spending Account work? ",
     "ground_truth": "A medical Flexible Spending Account (also known as a flexible spending arrangement or FSA) is a special account that an employee can put pre-tax money into to pay for certain out-of­pocket health care costs not covered by insurance. Employers may also make contributions to FSAs but aren't required to do so.\nFacts to keep in mind about FSAs include:\n• FSAs can only be offered in conjunction with an existing group healthcare plan.\n• There are two different kinds of FSAs an employer can offer. Health FSAs allow employees to pay for out-of­pocket medical expenses with tax-free dollars. Dependent Care FSAs allow employees to pay for expenses with tax­free dollars. Employers can offer one or both types to their employees.\n• Flexible spending accounts can cover expenses like daycare, prescriptions, dental work and eye exams.\n• Employees are reimbursed for expenses with a health flexible spending account, which means bills have to be paid out of pocket and the employee has to file a claim to get reimbursed from the FSA.\n• Money contributed to a health care FSA has to be spent within the current plan year or forfeited (although employers can allow a portion of that amount to be carried over to the next plan year if they so choose).\n• Employees with a medical FSA have immediate access to the full amount they elected to contribute for the tax year (even if some or all of that amount hasn't actually been deposited yet).\n• Only full time employees are eligible to participate in a Flexible Spending Account. Part time employees (those who work less than 30 hours per week) aren't eligible for an FSA.\n• Normally, employees can't change their per pay period contribution amount once they've decided on it, but exceptions can be made for events Iike the birth of a child or the death of a spouse.\n• Employees have the option of using a debit card, also known as a Flexcard, to withdraw money directly from their FSA. Using a Flexcard has the advantage of keeping a record of all withdrawals in one place.\n• Employees under the age of 65 who spend money in their FSA on non-medical expenses have to pay income tax on that money plus a penalty (usually 20 percent of the amount spent). Employees 65 or older have to pay income tax on the non-medical expense amounts, but aren't charged a penalty.\n• There are dozens of items that qualify as medical expenses under an FSA.\n• The employer owns the FSA so employees who leave are generally no longer eligible to participate in their FSA(unless they elect continuation coverage) and they forfeit any amount remaining in their account.\n\nEmployers can choose to let participants roll some of the money in their FSA to the new plan year by using one of two options:\n• Set a limit of up to $500 and al low participants to roll that much of their unspent funds over to the new plan year.\n• Elect to provide a 2 ½ month grace period following the end of the plan year for participants to claim whatever funds are left in their account. When the grace period ends the remaining funds become the property of the employer.\n\nNote: IRS Publication 502 has a list of what medical items are and aren't covered by an FSA.",
     "question_type": "how"},
    {"question": "How can employees enrolled in a qualified benefit plan learn about the provisions in that plan?",
     "ground_truth": "The summary plan description provided to employees when they sign up for the benefit plan includes a detailed overview of the plan, how it works, what benefits it provides, any limitations and how to file a claim. Copies of the SPD can be obtained from the plan administrator.",
     "question_type": "how"},
    {"question": "Can HRAs make healthcare insurance less of a hassle for employers?",
     "ground_truth": "HRAs are not only a more flexible alternative to standard group health insurance plans - they also take most of the management burden off of the employer. Rather than the employer having to weigh healthcare risks and decide on coverages, the employees can choose the coverage that works best for them and their families.",
     "question_type": "if/can"},
    {"question": " Can small employers who want to provide group health care join with other employers to help lower costs?",
    "ground_truth": "In many cases, trade associations offer health insurance plans for small-business owners and their employees at lower rates. A given employer may only have a handful of employees, but combining with thousands of employees in the association provides considerable leverage. In addition, since the carrier issues a policy to the whole association, no individual member can have their coverage canceled unless the carrier cancels coverage for the entire association.",
    "question_type": "if/can"},
    {"question": "Can employers offer different types of paid leave to different groups of employees?",
     "ground_truth": "Yes, as long as long as the leave policy doesn't discriminate against a protected class of employees and it complies with federal and state laws. For example, employers often offer different amounts of paid leave time based on length of service, or they have different leave policies for full-time and part-time employees.",
     "question_type": "if/can"},
]


# Main evaluation loop
def main():
    print("Starting evaluation")
    model_option = "E5 Base"
    prompt_levels = ["short", "medium", "long"]
    model_name = "Llama-2-7B"

    results = {}

    for data in questions_data:
        question = data["question"]
        ground_truth = data["ground_truth"]
        print("Starting evaluation for question: ", question)

        combined_context = retrieve_context(question, model_option)

        if question not in results:
            results[question] = {}

        for level in prompt_levels:
            prompt = construct_prompt(question, combined_context, level)
            response = get_llm_response(prompt)
            evaluations = evaluate_metrics(ground_truth, response, bart_scorer)

            if level not in results[question]:
                results[question][level] = {}

            results[question][level][model_name] = {
                "prompt": prompt,
                "response": response,
                "evaluations": evaluations
            }

    output_file = "evaluation_results_llama7b.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

# Run the main function
main()