import torch
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def load_model(model_path):
    """
    Memuat model dan tokenizer untuk inference.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def find_context(question, dataset_file):
    """
    Mencari konteks berdasarkan pertanyaan dari dataset.
    """
    data = pd.read_csv(dataset_file)
    for _, row in data.iterrows():
        # Cek kecocokan pertanyaan tanpa memperhatikan kapitalisasi
        if question.lower() in row['question'].lower():
            return row['context']
    return None

def answer_question(question, context, model, tokenizer, max_length=512):
    """
    Menghasilkan jawaban dari model berdasarkan pertanyaan dan konteks yang diberikan.
    """
    if context is None:
        return "Pertanyaan tidak valid atau tidak relevan."

    # Tokenisasi input
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    # Dapatkan prediksi dari model
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Konversi jawaban ke teks
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    # Jika jawaban tidak valid atau kosong
    if answer.strip() == "":
        return "Maaf, saya tidak menemukan jawaban yang relevan."

    return answer

def inference_from_input(question, model, tokenizer, dataset_file):
    """
    Melakukan inference dengan mencari konteks berdasarkan pertanyaan.
    """
    context = find_context(question, dataset_file)
    answer = answer_question(question, context, model, tokenizer)
    return answer

if __name__ == "__main__":
    model_path = "./fine_tuned_model/"  # Path ke model yang sudah dilatih
    dataset_file = "../dataset/qa_dataset.csv"  # Path ke dataset

    # Memuat model dan tokenizer
    model, tokenizer = load_model(model_path)

    while True:
        question = input("Masukkan pertanyaan: ")
        if question.lower() == "exit":
            break

        # Proses inference dan tampilkan jawaban
        answer = inference_from_input(question, model, tokenizer, dataset_file)
        print(f"Jawaban: {answer}")
