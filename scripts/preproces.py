import pandas as pd
from transformers import AutoTokenizer

def preprocess_data(dataset_file, qa_model_name, text_gen_model_name, max_length=512):
    """
    Preprocessing dataset untuk QA dan Text Generation.

    Args:
    - dataset_file: Lokasi file CSV dengan kolom 'question', 'context', 'answers'.
    - qa_model_name: Nama model untuk tokenisasi QA.
    - text_gen_model_name: Nama model untuk tokenisasi Text Generation.
    - max_length: Panjang maksimal token.

    Returns:
    - qa_tokenized_data: List of dictionaries dengan tokenisasi dan posisi jawaban (untuk QA).
    - text_gen_tokenized_data: List of dictionaries dengan tokenisasi (untuk Text Generation).
    """
    data = pd.read_csv(dataset_file)    
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    text_gen_tokenizer = AutoTokenizer.from_pretrained(text_gen_model_name)

    # Preprocessing untuk QA
    qa_tokenized_data = []
    for _, row in data.iterrows():
        question = row['question']
        context = row['context']
        answer_text = row['answers']
        start_char = context.find(answer_text)
        if start_char == -1:
            continue
        end_char = start_char + len(answer_text)
        inputs = qa_tokenizer(
            question, context, add_special_tokens=True, max_length=max_length,
            padding="max_length", truncation=True, return_offsets_mapping=True
        )
        offsets = inputs["offset_mapping"]
        start_token, end_token = None, None
        for idx, (start, end) in enumerate(offsets):
            if start == start_char:
                start_token = idx
            if end == end_char:
                end_token = idx
                break
        if start_token is None or end_token is None:
            continue
        qa_tokenized_data.append({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'start_positions': start_token,
            'end_positions': end_token,
        })

    # Preprocessing untuk Text Generation
    text_gen_tokenized_data = []
    for _, row in data.iterrows():
        context = row['context']
        inputs = text_gen_tokenizer(context, max_length=max_length, padding="max_length", truncation=True)
        text_gen_tokenized_data.append({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })

    return qa_tokenized_data, text_gen_tokenized_data
