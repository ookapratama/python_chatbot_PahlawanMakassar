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

def add_prompt_to_dataset(question, target, dataset_file):
    """
    Menambahkan prompt dan target baru ke dataset CSV.
    """
    try:
        # Baca dataset
        data = pd.read_csv(dataset_file)

        # Buat DataFrame untuk baris baru
        new_row = pd.DataFrame([{'prompt': question, 'target': target}])

        # Gabungkan DataFrame lama dengan baris baru
        data = pd.concat([data, new_row], ignore_index=True)

        # Simpan kembali ke file
        data.to_csv(dataset_file, index=False)
        print("Prompt dan target baru berhasil ditambahkan ke dataset.")
        return True
    except Exception as e:
        print(f"Error saat menambahkan prompt ke dataset: {e}")
        return False


def find_prompt_from_text_gen_dataset(question, text_gen_dataset_file):
    """
    Cari target (jawaban) dari dataset berdasarkan prompt.
    """
    try:
        data = pd.read_csv(text_gen_dataset_file)
        for _, row in data.iterrows():
            if question.lower() in row['prompt'].lower():
                return row['target']
        return None
    except Exception as e:
        print(f"Error saat mencari target dari dataset: {e}")
        return None


def text_generation_inference_only(prompt, dataset_file, model_path, max_new_tokens=50):
    """
    Menghasilkan jawaban berdasarkan prompt di dataset atau melakukan inferensi text generation.
    """
    # Cari target dari dataset
    try:
        target = find_prompt_from_text_gen_dataset(prompt, dataset_file)
        if target:
            print("Prompt ditemukan di dataset. Mengembalikan target...")
            return target  # Jika ditemukan, langsung kembalikan target
    except Exception as e:
        print(f"Terjadi kesalahan saat mencari prompt di dataset: {e}")

    # Jika tidak ditemukan, gunakan text generation
    try:
        print("Prompt tidak ditemukan di dataset. Menggunakan model text generation...")
        # Load model dan tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Tambahkan pad_token jika belum ada
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenisasi prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

        # Lakukan inferensi text generation
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2,  # Menghindari pengulangan teks
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        print(f"Terjadi kesalahan saat inferensi text generation: {e}")
        return f"Terjadi kesalahan saat inferensi text generation: {e}"
