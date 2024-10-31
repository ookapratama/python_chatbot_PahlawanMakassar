import pandas as pd
from transformers import AutoTokenizer

def preprocess_data(dataset_file, model_name, max_length=512):
    """
    Preprocessing dataset untuk QA.

    Args:
    - dataset_file: Lokasi file CSV dengan kolom 'question', 'context', 'answers'.
    - model_name: Nama model untuk tokenisasi.
    - max_length: Panjang maksimal token.

    Returns:
    - tokenized_data: List of dictionaries dengan tokenisasi dan posisi jawaban.
    """
    # Load dataset dan tokenizer
    data = pd.read_csv(dataset_file)    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_data = []
    for _, row in data.iterrows():
        question = row['question']
        context = row['context']
        answer_text = row['answers']

        # Cari posisi start dan end dari jawaban dalam konteks asli
        start_char = context.find(answer_text)
        if start_char == -1:
            continue  # Lewati jika jawaban tidak ditemukan dalam context
        end_char = start_char + len(answer_text)

        # Tokenisasi question dan context
        inputs = tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )

        # Konversi posisi karakter ke indeks token
        offsets = inputs["offset_mapping"]
        start_token, end_token = None, None

        for idx, (start, end) in enumerate(offsets):
            if start == start_char:
                start_token = idx
            if end == end_char:
                end_token = idx
                break

        # Jika start_token atau end_token tidak ditemukan, skip contoh ini
        if start_token is None or end_token is None:
            continue

        # Siapkan contoh tokenized
        tokenized_example = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'start_positions': start_token,
            'end_positions': end_token,
        }
        tokenized_data.append(tokenized_example)

    return tokenized_data
