import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM

def load_qa_model(qa_model_path):
    """
    Memuat model dan tokenizer untuk QA.
    """
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path)
        tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
        print(f"Model QA berhasil dimuat dari {qa_model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error saat memuat model QA: {e}")
        return None, None

def load_text_gen_model(gen_model_path):
    """
    Memuat model dan tokenizer untuk Text Generation.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(gen_model_path)
        tokenizer = AutoTokenizer.from_pretrained(gen_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Model Text Generation berhasil dimuat dari {gen_model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error saat memuat model Text Generation: {e}")
        return None, None

def find_context_from_qa_dataset(question, qa_dataset_file):
    """
    Mencari konteks berdasarkan pertanyaan dari dataset QA.
    """
    try:
        data = pd.read_csv(qa_dataset_file)
        for _, row in data.iterrows():
            if question.lower() in row['question'].lower():
                return row['context']
        return None
    except Exception as e:
        print(f"Error saat mencari konteks dari QA dataset: {e}")
        return None

def find_prompt_from_text_gen_dataset(question, text_gen_dataset_file):
    """
    Mencari prompt berdasarkan pertanyaan dari dataset Text Generation.
    """
    try:
        data = pd.read_csv(text_gen_dataset_file)
        for _, row in data.iterrows():
            if question.lower() in row['prompt'].lower():
                return row['prompt']
        return None
    except Exception as e:
        print(f"Error saat mencari prompt dari Text Generation dataset: {e}")
        return None

def qa_inference(question, context, qa_model, qa_tokenizer, max_length=512):
    """
    Melakukan inferensi dengan model QA untuk jawaban spesifik.
    """
    try:
        inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = qa_model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)
        return answer.strip()
    except Exception as e:
        return f"Terjadi kesalahan saat inferensi QA: {e}"

def text_generation_inference(prompt, gen_model, gen_tokenizer, max_new_tokens=50):
    """
    Melakukan inferensi dengan model Text Generation untuk penjelasan lebih luas.
    """
    try:
        inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        outputs = gen_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=gen_tokenizer.pad_token_id,
            no_repeat_ngram_size=2,
        )
        generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        return f"Terjadi kesalahan saat inferensi Text Generation: {e}"

def text_generation_inference_only(model_path, dataset_file, max_new_tokens=50):
    """
    Fungsi untuk menjalankan inferensi Text Generation saja.
    """
    try:
        # Load model dan tokenizer
        print(f"Memuat model dari {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tambahkan pad_token jika belum ada
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model Text Generation berhasil dimuat dari {model_path}")

        # Loop inferensi
        while True:
            print("\nMasukkan 'exit' untuk keluar.")
            question = input("Masukkan pertanyaan atau prompt: ")
            if question.lower() == "exit":
                break

            # Cari prompt dari dataset (jika tersedia)
            prompt = find_prompt_from_text_gen_dataset(question, dataset_file)
            if not prompt:
                prompt = question  # Gunakan input pengguna langsung sebagai prompt

            # Lakukan inferensi Text Generation
            print(f"Menghasilkan teks berdasarkan prompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"\nTeks yang dihasilkan:\n{generated_text.strip()}")
    
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")


if __name__ == "__main__":
    # Path ke model
    gen_model_path = "./tuning_textGen_nusantara_indo_chat/"
    # gen_model_path = "./tuning_gpt2_medium_indo/"
    text_gen_dataset_file = "../dataset/text_generation_dataset.csv"

    print("Pilih mode inferensi:")
    print("1. Text Generation saja")
    print("2. QA dan Text Generation")
    mode = input("Masukkan pilihan (1/2): ")

    if mode == "1":
        # Jalankan fungsi Text Generation saja
        text_generation_inference_only(gen_model_path, text_gen_dataset_file)
    elif mode == "2":
        # Jalankan QA dan Text Generation (fungsi sebelumnya)
        qa_model_path = "./tuning_qa/"
        qa_dataset_file = "../dataset/enhanced_qa_dataset.csv"
        qa_model, qa_tokenizer = load_qa_model(qa_model_path)
        gen_model, gen_tokenizer = load_text_gen_model(gen_model_path)

        if qa_model is None or qa_tokenizer is None:
            print("Model atau tokenizer QA gagal dimuat.")
            exit()

        if gen_model is None or gen_tokenizer is None:
            print("Model atau tokenizer Text Generation gagal dimuat.")
            exit()

        while True:
            print("\nMasukkan 'exit' untuk keluar.")
            question = input("Masukkan pertanyaan: ")
            if question.lower() == "exit":
                break

            # Cari konteks otomatis dari dataset QA
            context = find_context_from_qa_dataset(question, qa_dataset_file)
            if not context:
                print("Konteks tidak ditemukan dalam dataset QA. Harap masukkan konteks secara manual.")
                context = input("Masukkan konteks: ")
                if context.lower() == "exit":
                    break

            # Cari prompt otomatis dari dataset Text Generation
            prompt = find_prompt_from_text_gen_dataset(question, text_gen_dataset_file)
            if not prompt:
                prompt = f"Pertanyaan: {question}\nKonteks: {context}"

            # Jawaban spesifik dengan model QA
            specific_answer = qa_inference(question, context, qa_model, qa_tokenizer)

            # Penjelasan lebih luas dengan model Text Generation
            detailed_answer = text_generation_inference(prompt, gen_model, gen_tokenizer)

            # Output jawaban
            print(f"\nJawaban spesifik (QA): {specific_answer}")
            print(f"Penjelasan lebih luas (Text Generation): {detailed_answer}")
    else:
        print("Pilihan tidak valid. Keluar dari program.")
