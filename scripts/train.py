import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset

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

# import pandas as pd
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# import pandas as pd
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def train_text_generation(model_name, dataset_file, output_dir, max_length=512):
    """
    Melatih model Text Generation.
    """
    # Load dataset
    data = pd.read_csv(dataset_file)

    # Validasi data
    data['prompt'] = data['prompt'].astype(str)
    data['target'] = data['target'].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenisasi dataset (gunakan batch tokenization)
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

    # Konversi dataset ke Dataset dari Hugging Face
    dataset = Dataset.from_pandas(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs_text_gen",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model Text Generation berhasil disimpan di {output_dir}")
