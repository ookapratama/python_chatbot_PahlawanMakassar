import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering

def preprocess_data_for_qa(dataset_file, model_name, max_length=512):
    data = pd.read_csv(dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_data = []
    for _, row in data.iterrows():
        question = row['question']
        context = row['context']
        answer = row['answers']

        inputs = tokenizer(
            question, context, add_special_tokens=True, max_length=max_length, truncation=True, return_offsets_mapping=True
        )
        start_char = context.find(answer)
        end_char = start_char + len(answer)

        start_token, end_token = None, None
        for idx, (start, end) in enumerate(inputs["offset_mapping"]):
            if start == start_char:
                start_token = idx
            if end == end_char:
                end_token = idx
                break

        if start_token is None or end_token is None:
            continue

        tokenized_example = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'start_positions': start_token,
            'end_positions': end_token,
        }
        tokenized_data.append(tokenized_example)

    return tokenized_data

def prepare_text_generation_dataset(dataset_file, qa_model_path, output_file, max_length=512):
    data = pd.read_csv(dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path)

    text_gen_data = []
    for _, row in data.iterrows():
        question = row['question']
        context = row['context']

        inputs = tokenizer(
            question, context, return_tensors="pt", truncation=True, max_length=max_length
        )
        outputs = qa_model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1

        answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx], skip_special_tokens=True)

        prompt = f"{question} {context}"
        target = answer
        text_gen_data.append({"prompt": prompt, "target": target})

    pd.DataFrame(text_gen_data).to_csv(output_file, index=False)
    print(f"Dataset Text Generation disimpan di {output_file}")
