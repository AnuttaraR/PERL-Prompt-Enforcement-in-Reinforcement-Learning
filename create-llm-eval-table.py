import json
import pandas as pd

# Load the JSON data
with open('evaluation_results_gpt3_gpt4_all_metrics.json', 'r') as f:
    data = json.load(f)

# Initialize an empty list to hold all records
records = []

# Iterate over the data and extract information for both GPT-3 and GPT-4
for question, prompt_levels in data.items():
    for prompt_level, models in prompt_levels.items():
        # Extract GPT-3 and GPT-4 information (assuming both are always present)
        gpt3_data = models.get('GPT-3', {})
        gpt4_data = models.get('GPT-4', {})

        # Prepare a single record that includes both GPT-3 and GPT-4 columns side-by-side
        record = {
            "Question": question,
            "GPT-3 Prompt Level": prompt_level,
            "GPT-3 Prompt": gpt3_data.get("prompt", ""),
            "GPT-3 Response": gpt3_data.get("response", ""),
            "GPT-3 BLEU": gpt3_data.get("evaluations", {}).get("BLEU", ""),
            "GPT-3 ROUGE-1": gpt3_data.get("evaluations", {}).get("ROUGE", {}).get("rouge1", ""),
            "GPT-3 ROUGE-2": gpt3_data.get("evaluations", {}).get("ROUGE", {}).get("rouge2", ""),
            "GPT-3 ROUGE-L": gpt3_data.get("evaluations", {}).get("ROUGE", {}).get("rougeL", ""),
            "GPT-3 METEOR": gpt3_data.get("evaluations", {}).get("METEOR", ""),
            "GPT-3 BERT Precision": gpt3_data.get("evaluations", {}).get("BERTScore", {}).get("Precision", ""),
            "GPT-3 BERT Recall": gpt3_data.get("evaluations", {}).get("BERTScore", {}).get("Recall", ""),
            "GPT-3 BERT F1": gpt3_data.get("evaluations", {}).get("BERTScore", {}).get("F1", ""),
            "GPT-3 BARTScore": gpt3_data.get("evaluations", {}).get("BARTScore", ""),
            "GPT-4 Prompt Level": prompt_level,
            "GPT-4 Prompt": gpt4_data.get("prompt", ""),
            "GPT-4 Response": gpt4_data.get("response", ""),
            "GPT-4 BLEU": gpt4_data.get("evaluations", {}).get("BLEU", ""),
            "GPT-4 ROUGE-1": gpt4_data.get("evaluations", {}).get("ROUGE", {}).get("rouge1", ""),
            "GPT-4 ROUGE-2": gpt4_data.get("evaluations", {}).get("ROUGE", {}).get("rouge2", ""),
            "GPT-4 ROUGE-L": gpt4_data.get("evaluations", {}).get("ROUGE", {}).get("rougeL", ""),
            "GPT-4 METEOR": gpt4_data.get("evaluations", {}).get("METEOR", ""),
            "GPT-4 BERT Precision": gpt4_data.get("evaluations", {}).get("BERTScore", {}).get("Precision", ""),
            "GPT-4 BERT Recall": gpt4_data.get("evaluations", {}).get("BERTScore", {}).get("Recall", ""),
            "GPT-4 BERT F1": gpt4_data.get("evaluations", {}).get("BERTScore", {}).get("F1", ""),
            "GPT-4 BARTScore": gpt4_data.get("evaluations", {}).get("BARTScore", ""),
        }

        # Append the record to the list of records
        records.append(record)

# Convert the list of records to a pandas DataFrame
df = pd.DataFrame(records)

# Save the DataFrame to an Excel file with a nicely formatted structure
output_file = 'D:/evaluation_comparison_gpt3_gpt4_all_metrics.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Comparisons', index=False)

    # Get the xlsxwriter objects from the DataFrame writer object
    workbook  = writer.book
    worksheet = writer.sheets['Comparisons']

    # Define the formats for the headers (merged cells for GPT-3 and GPT-4)
    merge_format = workbook.add_format({
        'bold': True,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })

    # Merge cells for GPT-3 and GPT-4 headers
    worksheet.merge_range('B1:L1', 'GPT-3', merge_format)
    worksheet.merge_range('M1:W1', 'GPT-4', merge_format)

# Notify the user
print(f"Data has been successfully exported to {output_file}")
