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

    # Tokenisasi dataset
    dataset = Dataset.from_dict({
        'input_ids': [
            tokenizer(prompt, max_length=max_length, truncation=True, return_tensors="pt")['input_ids'][0]
            for prompt in data['prompt']
        ],
        'labels': [
            tokenizer(target, max_length=max_length, truncation=True, return_tensors="pt")['input_ids'][0]
            for target in data['target']
        ],
    })

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
        train_dataset=dataset
    )
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model Text Generation berhasil disimpan di {output_dir}")
