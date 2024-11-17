import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """
    Memuat model dan tokenizer untuk Text Generation.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Model dan tokenizer berhasil dimuat dari {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error saat memuat model atau tokenizer: {e}")
        return None, None

def find_context(question, dataset_file):
    """
    Mencari konteks berdasarkan pertanyaan dari dataset.
    """
    try:
        data = pd.read_csv(dataset_file)
        for _, row in data.iterrows():
            # Cek kecocokan pertanyaan (case-insensitive)
            if question.lower() in row['question'].lower():
                return row['context']
        return None
    except Exception as e:
        print(f"Error saat mencari konteks: {e}")
        return None

def inference_text_generation(question, context, model, tokenizer, max_length=50):
    """
    Melakukan inferensi dengan model Text Generation.
    """
    if model is None or tokenizer is None:
        return "Model atau tokenizer belum dimuat dengan benar."

    # Gabungkan pertanyaan dan konteks menjadi prompt
    prompt = f"{question} {context}" if context else f"{question}"
    
    try:
        # Tokenisasi prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        
        # Hasilkan teks
        outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text.strip()
    except Exception as e:
        return f"Terjadi kesalahan saat inferensi: {e}"

if __name__ == "__main__":
    # Path ke model Text Generation yang telah dilatih
    model_path = "./fine_tuned_model_text_gen/"
    dataset_file = "../dataset/enhanced_qa_dataset.csv"  # Dataset untuk mencari konteks
    
    # Memuat model dan tokenizer
    model, tokenizer = load_model(model_path)

    # Loop inferensi
    while True:
        print("\nMasukkan 'exit' untuk keluar.")
        question = input("Masukkan pertanyaan: ")
        if question.lower() == "exit":
            break
        
        # Cari konteks otomatis dari dataset
        context = find_context(question, dataset_file)
        if not context:
            print("Konteks tidak ditemukan dalam dataset. Harap masukkan konteks secara manual.")
            context = input("Masukkan konteks: ")
            if context.lower() == "exit":
                break

        # Proses inferensi
        answer = inference_text_generation(question, context, model, tokenizer, max_length=50)
        print(f"\nJawaban: {answer}")
