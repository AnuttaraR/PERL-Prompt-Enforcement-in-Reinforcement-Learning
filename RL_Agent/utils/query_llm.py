import openai
from openai import OpenAI
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Logging Configuration
logging.basicConfig(
    level=logging.INFO
)

# Load OpenAI API key from environment (Set before running)
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# Load models for LLaMA and Mistral
device = "cuda" if torch.cuda.is_available() else "cpu"

# try:
#     logging.info("Loading LLaMA-2 model...")
#     llama_tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-32K-Instruct-AWQ", trust_remote_code=True)
#     llama_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-32K-Instruct-AWQ",
#                                                        trust_remote_code=True).to(device)
#     logging.info("LLaMA-2 model loaded successfully.")
# except Exception as e:
#     logging.error(f"Failed to load LLaMA-2 model: {e}")
#
# try:
#     logging.info("Loading Mistral model...")
#     mistral_tokenizer = AutoTokenizer.from_pretrained("alokabhishek/Mistral-7B-Instruct-v0.2-GGUF")
#     mistral_model = AutoModelForCausalLM.from_pretrained(
#         "alokabhishek/Mistral-7B-Instruct-v0.2-GGUF",
#         model_file="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
#         model_type="mistral",
#         gpu_layers=50,
#         hf=True
#     ).to(device)
#     logging.info("Mistral model loaded successfully.")
# except Exception as e:
#     logging.error(f"Failed to load Mistral model: {e}")


def get_llm_response(prompt, model, tokenizer=None, max_tokens=500):
    """Query the specified LLM model and log response status."""
    # 🛠️ Ensure `prompt` is a string before sending it to LLM
    if isinstance(prompt, dict) and "input_ids" in prompt:
        prompt = tokenizer.decode(prompt["input_ids"], skip_special_tokens=True)

    logging.info(f"🚀 LLM CALL 1: Actual Prompt Sent to {model}: {prompt}")
    try:
        if model in ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at editing prompts and following instructions given"
                                                  " to edit the prompts. You must follow the instructions given to you for"
                                                  " editing a given prompt **directly** and **concisely**. Do not ask questions,"
                                                  " Do not go to answer the question given in the prompt, your task is to only"
                                                  " edit the prompt given according to the information. Only response expected is"
                                                  " the edited prompt according to the instructions and nothing else. Do not say 'I need more"
                                                  " information'. If the input is unclear, respond with 'Invalid input'"},
                    {"role": "user", "content": prompt + "\nAdjusted Prompt:"},
                ],
                temperature=0.7,
            )

            response_message = response.choices[0].message.content

            print("LLM CALL 1 Response: ", response_message)

            INVALID_RESPONSE = "INVALID"

            if response_message.lower() in ["invalid input.", "invalid response.", "error.", "invalid", "invalid input", "invalid response", "error"]:
                logging.warning(f"⚠️ LLM returned '{INVALID_RESPONSE}'—resetting to original question.")
                return INVALID_RESPONSE

            logging.info(f"✔️ LLM CALL 1 DONE: Prompt + Action Before Modification: {prompt},\n Modified Prompt: {response_message}")

            return response_message

        # elif model == "llama-2":
        #     # LLaMA-2 model
        #     inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        #     outputs = llama_model.generate(inputs["input_ids"], num_return_sequences=1, max_new_tokens=max_tokens)
        #     output = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        #     logging.info(f"LLaMA-2 Response: {output[:150]}...")  # Log first 150 chars
        #     return output
        #
        # elif model == "mistral":
        #     # Mistral model
        #     inputs = mistral_tokenizer(prompt, return_tensors="pt", max_length=512).to(device)
        #     generate_ids = mistral_model.generate(inputs.input_ids, max_new_tokens=max_tokens).to(device)
        #     output = mistral_tokenizer.decode(generate_ids[0], skip_special_tokens=True,
        #                                       clean_up_tokenization_spaces=False).strip()
        #     logging.info(f"Mistral Response: {output[:150]}...")  # Log first 150 chars
        #     return output

        else:
            logging.error(f"Invalid model specified: {model}")
            raise ValueError("Invalid model specified. Choose from 'gpt-4', 'gpt-3.5-turbo', 'llama-2', or 'mistral'.")

    except Exception as e:
        logging.error(f"Error querying model '{model}': {e}")
        return "Error in generating response."


def generate_answer_from_llm(prompt, model, max_tokens=500):
    """Query the specified LLM model to generate an actual answer to a given prompt."""
    print(f"📝 LLM CALL 2: Generating final answer for prompt: {prompt[:50]}")

    try:
        if model in ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            response_message = response.choices[0].message.content.strip()
            print(f"✅ LLM CALL 2: Generated Answer from Modified Prompt: {response_message[:50]}...")

            return response_message

        else:
            print(f"Invalid model specified: {model}")
            raise ValueError("Invalid model specified. Choose from 'gpt-4' or 'gpt-3.5-turbo'.")

    except Exception as e:
        print(f"Error querying model '{model}': {e}")
        return "Error in generating answer."


# Test function to verify it works
if __name__ == "__main__":
    test_prompt = "Explain the concept of reinforcement learning in simple terms."

    print("\nGPT-4 Response:")
    print(get_llm_response(test_prompt, model="gpt-3.5-turbo"))

    # print("\nGPT-3.5 Turbo Response:")
    # print(get_llm_response(test_prompt, model="gpt-3.5-turbo"))
    #
    # print("\nLLaMA-2 Response:")
    # print(get_llm_response(test_prompt, model="llama-2"))
    #
    # print("\nMistral Response:")
    # print(get_llm_response(test_prompt, model="mistral"))
