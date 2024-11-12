import torch
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from preproces import preprocess_data  # Pastikan nama file preprocess.py sesuai
from datasets import Dataset

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

    # Load model dan tokenizer
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

if __name__ == "__main__":
    # model_name = "indobenchmark/indobert-base-p1"  # IndoBERT untuk bahasa Indonesia
    # model_name = "indolem/indobertqa-base"  # IndoBERT untuk bahasa Indonesia
    # model_name = "mrm8488/bert-small-finetuned-squadv2"  # IndoBERT untuk bahasa Indonesia
    # model_name = "Wikidepia/indobert-lite-squad"  # IndoBERT untuk bahasa Indonesia
    model_name = "malaputri/indobert-squad-id"  
    # dataset_file = "../dataset/qa_dataset.csv"  # Dataset yang sudah disesuaikan
    dataset_file = "../dataset/enhanced_qa_dataset.csv"  # Dataset yang sudah disesuaikan
    output_dir = "./fine_tuned_model/"

    train_model(model_name, dataset_file, output_dir)
