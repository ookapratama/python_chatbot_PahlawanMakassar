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

def train_text_generation(model_name, dataset_file, output_dir, max_length=512):
    data = pd.read_csv(dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir="./logs_text_gen",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
