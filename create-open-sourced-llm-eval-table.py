import json
import pandas as pd

# Load JSON data
with open('evaluation_results_mistral_128k.json', 'r') as file:
    data = json.load(file)

# Flatten the JSON structure
flattened_data = []

for question, levels in data.items():
    for level, details in levels.items():
        row = {
            "Question": question,
            "Prompt Level": level,
            "Prompt": details["prompt"],
            "Response": details["response"],
            "BLEU": details["evaluations"]["BLEU"],
            "ROUGE_1_Precision": details["evaluations"]["ROUGE"]["rouge1"][0],
            "ROUGE_1_Recall": details["evaluations"]["ROUGE"]["rouge1"][1],
            "ROUGE_1_F1": details["evaluations"]["ROUGE"]["rouge1"][2],
            "ROUGE_2_Precision": details["evaluations"]["ROUGE"]["rouge2"][0],
            "ROUGE_2_Recall": details["evaluations"]["ROUGE"]["rouge2"][1],
            "ROUGE_2_F1": details["evaluations"]["ROUGE"]["rouge2"][2],
            "ROUGE_L_Precision": details["evaluations"]["ROUGE"]["rougeL"][0],
            "ROUGE_L_Recall": details["evaluations"]["ROUGE"]["rougeL"][1],
            "ROUGE_L_F1": details["evaluations"]["ROUGE"]["rougeL"][2],
            "METEOR": details["evaluations"]["METEOR"],
            "BERTScore_Precision": details["evaluations"]["BERTScore"]["Precision"],
            "BERTScore_Recall": details["evaluations"]["BERTScore"]["Recall"],
            "BERTScore_F1": details["evaluations"]["BERTScore"]["F1"],
            "BARTScore": details["evaluations"]["BARTScore"]
        }
        flattened_data.append(row)

# Create DataFrame
df = pd.DataFrame(flattened_data)

# Save to Excel
df.to_excel("D:/evaluation_results_mistral_7b_128k.xlsx", index=False)
print("Data has been saved to evaluation_results.xlsx")
