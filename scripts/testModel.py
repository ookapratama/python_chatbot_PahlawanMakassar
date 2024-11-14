import torch
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from preproces import preprocess_data  # Pastikan nama file preprocess.py sesuai
from datasets import Dataset
import pandas as pd

def train_model(model_name, dataset_file, output_dir, max_length=512):
    """
    Melatih model QA dengan dataset yang dipreproses.

    Args:
    - model_name: Nama model yang akan dilatih.
    - dataset_file: Lokasi dataset CSV.
    - output_dir: Lokasi untuk menyimpan model yang sudah dilatih.
    - max_length: Panjang maksimal token.
    """
    # Preprocess data dan konversi menjadi Dataset dari Hugging Face
    tokenized_data = preprocess_data(dataset_file, model_name, max_length)
    print("Contoh data tokenized dari preprocess:")
    for i in range(2):  # Menampilkan 2 contoh pertama
        print(tokenized_data[i])

    # Konversi hasil tokenisasi menjadi format Dataset yang bisa digunakan oleh Trainer
    dataset = Dataset.from_dict({
        'input_ids': [data['input_ids'] for data in tokenized_data],
        'attention_mask': [data['attention_mask'] for data in tokenized_data],
        'start_positions': [data['start_positions'] for data in tokenized_data],
        'end_positions': [data['end_positions'] for data in tokenized_data]
    }).with_format("torch")  # Konversi ke tensor

    # Load model dan tokenizer untuk QA
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",  # Lokasi untuk menyimpan log
        logging_steps=10,
        save_steps=500
    )

    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer
    )

    # Melatih model
    trainer.train()

    # Simpan model yang sudah dilatih
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# Fungsi untuk memuat konteks dari dataset
def load_context_from_dataset(dataset_file):
    df = pd.read_csv(dataset_file)
    context = df['context'][0]  # Ambil konteks dari baris pertama, sesuaikan jika ingin yang lain
    return context

# Fungsi untuk menggunakan model text generation berdasarkan jawaban dari QA
def generate_text_with_model(prompt, tg_model, tg_tokenizer, max_length=100):
    inputs = tg_tokenizer(prompt, return_tensors="pt")
    outputs = tg_model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tg_tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # QA model configuration
    model_name = "malaputri/indobert-squad-id"  
    dataset_file = "../dataset/enhanced_qa_dataset.csv"  # Dataset yang sudah disesuaikan
    output_dir = "./fine_tuned_model/"

    # Train QA model
    qa_model, qa_tokenizer = train_model(model_name, dataset_file, output_dir)

    # Load context from dataset
    context = load_context_from_dataset(dataset_file)
    print("Loaded Context:", context)

    # Load Text Generation Model
    tg_model_name = "Xenova/gpt-3.5-turbo"  # Model text generation
    tg_tokenizer = AutoTokenizer.from_pretrained(tg_model_name)
    tg_model = AutoModelForCausalLM.from_pretrained(tg_model_name)

    # Input pertanyaan dari pengguna
    question = input("Masukkan pertanyaan Anda: ")

    # Jawab pertanyaan dengan model QA
    inputs = qa_tokenizer(question, context, return_tensors="pt")
    qa_outputs = qa_model(**inputs)
    answer_start = torch.argmax(qa_outputs.start_logits)
    answer_end = torch.argmax(qa_outputs.end_logits) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
    )

    print("Jawaban dari Model QA:", answer)

    # Generate text based on the answer
    generated_text = generate_text_with_model(answer, tg_model, tg_tokenizer)
    print("Generated Text from Text Generation Model:", generated_text)
