import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForCausalLM, AutoTokenizer

def load_models(qa_model_path, text_gen_model_path):
    """
    Memuat model dan tokenizer untuk Question Answering dan Text Generation.
    """
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path)
    text_gen_model = AutoModelForCausalLM.from_pretrained(text_gen_model_path)
    
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
    text_gen_tokenizer = AutoTokenizer.from_pretrained(text_gen_model_path)
    
    return (qa_model, qa_tokenizer), (text_gen_model, text_gen_tokenizer)

def answer_question(question, context, model, tokenizer, max_length=512):
    """
    Menghasilkan jawaban dari model QA berdasarkan pertanyaan dan konteks yang diberikan.
    """
    if context is None:
        return "Pertanyaan tidak valid atau tidak relevan."
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="pt",
        max_length=max_length, padding="max_length", truncation=True
    )
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer if answer.strip() else "Maaf, saya tidak menemukan jawaban yang relevan."

def generate_text(prompt, model, tokenizer, max_length=50):
    """
    Menghasilkan teks dari model Text Generation berdasarkan prompt yang diberikan.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def inference_both_models(question, qa_model, qa_tokenizer, text_gen_model, text_gen_tokenizer, dataset_file):
    """
    Melakukan inferensi pada kedua model, Question Answering dan Text Generation.
    """
    context = find_context(question, dataset_file)
    qa_answer = answer_question(question, context, qa_model, qa_tokenizer)
    generated_text = generate_text(question, text_gen_model, text_gen_tokenizer)
    
    return qa_answer, generated_text

def find_context(question, dataset_file):
    """
    Mencari konteks berdasarkan pertanyaan dari dataset.
    """
    import pandas as pd
    data = pd.read_csv(dataset_file)
    for _, row in data.iterrows():
        if question.lower() in row['question'].lower():
            return row['context']
    return None

if __name__ == "__main__":
    # Path model yang sudah dilatih
    qa_model_path = "./fine_tuned_model_qa/"
    text_gen_model_path = "./fine_tuned_model_text_gen/"
    dataset_file = "../dataset/enhanced_qa_dataset.csv"  # Path ke dataset

    # Memuat kedua model dan tokenizernya
    (qa_model, qa_tokenizer), (text_gen_model, text_gen_tokenizer) = load_models(qa_model_path, text_gen_model_path)

    while True:
        question = input("Masukkan pertanyaan atau prompt: ")
        if question.lower() == "exit":
            break

        # Proses inferensi pada kedua model dan tampilkan hasilnya
        qa_answer, generated_text = inference_both_models(
            question, qa_model, qa_tokenizer, text_gen_model, text_gen_tokenizer, dataset_file
        )
        print(f"Jawaban QA: {qa_answer}")
        print(f"Teks yang dihasilkan: {generated_text}")
