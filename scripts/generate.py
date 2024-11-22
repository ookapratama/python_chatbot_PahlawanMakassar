import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def train_text_generation(model_name, dataset_file, output_dir, max_length=512):
    """
    Melatih model Text Generation.
    """
    try:
        # Step 1: Load dataset
        data = pd.read_csv(dataset_file)

        # Step 2: Validasi dataset
        if 'prompt' not in data.columns or 'target' not in data.columns:
            raise ValueError("Dataset harus memiliki kolom 'prompt' dan 'target'.")
        print("Dataset berhasil dimuat.")

        # Bersihkan data: hapus nilai kosong dan pastikan kolom berisi string
        data.dropna(subset=['prompt', 'target'], inplace=True)
        data['prompt'] = data['prompt'].astype(str).str.strip()
        data['target'] = data['target'].astype(str).str.strip()

        # Step 3: Split dataset menjadi training dan evaluasi
        train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
        print(f"Dataset dibagi menjadi {len(train_data)} data training dan {len(eval_data)} data evaluasi.")

        # Konversi ke Dataset Hugging Face
        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)

        # Step 4: Load tokenizer dan model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Tokenizer dan model berhasil dimuat.")

        # Fungsi untuk tokenisasi
        def tokenize_function(examples):
            try:
                inputs = tokenizer(
                    examples["prompt"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                labels = tokenizer(
                    examples["target"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                inputs["labels"] = labels["input_ids"]
                return inputs
            except Exception as e:
                print(f"Error saat tokenisasi data: {examples}")
                raise e

        # Tokenisasi dataset
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        print("Dataset berhasil ditokenisasi.")

        # Step 5: Set training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",  # Evaluasi di akhir setiap epoch
            save_strategy="epoch",  # Simpan model di akhir setiap epoch
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_dir="./logs_text_gen",
            logging_steps=10,
            save_total_limit=2,  # Simpan hanya 2 checkpoint terakhir
        )

        # Step 6: Setup Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
        )

        # Step 7: Latih model
        print("Mulai melatih model...")
        trainer.train()

        # Step 8: Simpan model dan tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model Text Generation berhasil disimpan di {output_dir}")

    except Exception as e:
        print(f"Terjadi error selama proses training: {e}")

# Main function untuk menjalankan training
if __name__ == "__main__":
    # Parameter
    model_name = "TurkuNLP/gpt3-finnish-small"  # Nama model
    dataset_file = "../dataset/text_gen.csv"  # Path ke dataset
    output_dir = "./fine_tuned_model_text_gen"  # Path output
    max_length = 512  # Panjang maksimum token

    # Jalankan fungsi training
    train_text_generation(model_name, dataset_file, output_dir, max_length)