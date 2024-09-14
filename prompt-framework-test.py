import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate, ManualVerbalizer

# Initialize tokenizer, model, and template
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)

# Define your template
template_text = '{"placeholder":"text_a"} It was {"mask"}'
template = ManualTemplate(text=template_text, tokenizer=tokenizer)

# Define your verbalizer
verbalizer = ManualVerbalizer(
    classes=['negative', 'positive'],
    label_words={'negative': ['bad'], 'positive': ['good', 'wonderful', 'great']},
    tokenizer=tokenizer
)

# Example data
train_data = [
    {"text_a": "I love this movie", "label": 1},  # Positive
    {"text_a": "I hate this movie", "label": 0}  # Negative
]

# Convert data to InputExample format
train_examples = [InputExample(guid=str(i), text_a=item['text_a'], label=item['label']) for i, item in
                  enumerate(train_data)]

# Create PromptDataLoader
train_dataset = PromptDataLoader(
    dataset=train_examples,
    template=template,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=None,
    max_seq_length=64,
    batch_size=4,
    shuffle=True
)

# Define training arguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Training
trainer.train()


# Inference
def infer(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_token_ids = torch.argmax(logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0].tolist())
    return predicted_tokens


# Test inference
test_text = "I love this [MASK]"
print(infer(test_text))
