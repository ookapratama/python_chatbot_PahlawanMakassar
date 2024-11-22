import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split


def train_qa(model_name, dataset_file, output_dir, max_length=512):
    from preproces import preprocess_data_for_qa
    tokenized_data = preprocess_data_for_qa(dataset_file, model_name, max_length)
    dataset = Dataset.from_dict({
        'input_ids': [d['input_ids'] for d in tokenized_data],
        'attention_mask': [d['attention_mask'] for d in tokenized_data],
        'start_positions': [d['start_positions'] for d in tokenized_data],
        'end_positions': [d['end_positions'] for d in tokenized_data],
    })

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir="./logs_qa",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)





def train_text_generation(model_name, dataset_file, output_dir, max_length=512):
    """
    Melatih model Text Generation dengan GPT-2.
    """
    try:
        # Step 1: Load dataset
        data = pd.read_csv(dataset_file)

        # Step 2: Validasi dataset
        if 'prompt' not in data.columns or 'target' not in data.columns:
            raise ValueError("Dataset harus memiliki kolom 'prompt' dan 'target'.")

        # Bersihkan data
        data.dropna(subset=['prompt', 'target'], inplace=True)
        data['prompt'] = data['prompt'].astype(str).str.strip()
        data['target'] = data['target'].astype(str).str.strip()

        # Bagi dataset menjadi training dan evaluasi
        train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
        print(f"Dataset dibagi menjadi {len(train_data)} data training dan {len(eval_data)} data evaluasi.")

        # Konversi ke Dataset Hugging Face
        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)

        # Step 3: Load tokenizer dan model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Tambahkan token padding jika belum ada
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer berhasil dimuat dan padding token disesuaikan.")

        # Step 4: Tokenisasi dataset
        def tokenize_function(examples):
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

        # Tokenisasi dataset
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Step 5: Set training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,  # Memuat model terbaik di akhir pelatihan
            metric_for_best_model="eval_loss",  # Metrik untuk menentukan model terbaik
            greater_is_better=False,  # Karena lebih rendah lebih baik untuk eval_loss
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_dir="./logs_text_gen",
            save_total_limit=2,  # Simpan hanya 2 checkpoint terakhir
            save_strategy="epoch",
            logging_steps=10,
        )

        # Step 6: Setup Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,  # Tambahkan eval_dataset
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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