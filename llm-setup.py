from gpt4all import GPT4All

model = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf")
with model.chat_session():
    print(model.generate("What is deductible and oop max in terms of insurance?",  max_tokens=1024))