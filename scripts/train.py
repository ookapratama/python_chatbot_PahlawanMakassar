import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForCausalLM, Trainer, TrainingArguments
from preproces import preprocess_data
from datasets import Dataset

def train_models(qa_model_name, text_gen_model_name, dataset_file, output_dir_qa, output_dir_text_gen, max_length=512):
    """
    Melatih model QA dan Text Generation dengan dataset yang dipreproses.

    Args:
    - qa_model_name: Nama model untuk Question Answering.
    - text_gen_model_name: Nama model untuk Text Generation.
    - dataset_file: Lokasi dataset CSV.
    - output_dir_qa: Lokasi untuk menyimpan model QA.
    - output_dir_text_gen: Lokasi untuk menyimpan model Text Generation.
    - max_length: Panjang maksimal token.
    """
    qa_tokenized_data, text_gen_tokenized_data = preprocess_data(dataset_file, qa_model_name, text_gen_model_name, max_length)

    # Training untuk model QA
    qa_dataset = Dataset.from_dict({
        'input_ids': [data['input_ids'] for data in qa_tokenized_data],
        'attention_mask': [data['attention_mask'] for data in qa_tokenized_data],
        'start_positions': [data['start_positions'] for data in qa_tokenized_data],
        'end_positions': [data['end_positions'] for data in qa_tokenized_data]
    }).with_format("torch")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

    qa_training_args = TrainingArguments(
        output_dir=output_dir_qa, evaluation_strategy="epoch",
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        num_train_epochs=3, weight_decay=0.01, logging_dir="./logs_qa"
    )

    qa_trainer = Trainer(model=qa_model, args=qa_training_args, train_dataset=qa_dataset, eval_dataset=qa_dataset)
    qa_trainer.train()
    qa_model.save_pretrained(output_dir_qa)

    # Training untuk model Text Generation
    text_gen_dataset = Dataset.from_dict({
        'input_ids': [data['input_ids'] for data in text_gen_tokenized_data],
        'attention_mask': [data['attention_mask'] for data in text_gen_tokenized_data]
    }).with_format("torch")
    text_gen_model = AutoModelForCausalLM.from_pretrained(text_gen_model_name)

    text_gen_training_args = TrainingArguments(
        output_dir=output_dir_text_gen, evaluation_strategy="epoch",
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        num_train_epochs=3, weight_decay=0.01, logging_dir="./logs_text_gen"
    )

    text_gen_trainer = Trainer(model=text_gen_model, args=text_gen_training_args, train_dataset=text_gen_dataset, eval_dataset=text_gen_dataset)
    text_gen_trainer.train()
    text_gen_model.save_pretrained(output_dir_text_gen)

if __name__ == "__main__":
    qa_model_name = "malaputri/indobert-squad-id"
    text_gen_model_name = "TurkuNLP/gpt3-finnish-small"  # Contoh untuk Text Generation
    dataset_file = "../dataset/enhanced_qa_dataset.csv"
    output_dir_qa = "./fine_tuned_model_qa/"
    output_dir_text_gen = "./fine_tuned_model_text_gen/"
    
    train_models(qa_model_name, text_gen_model_name, dataset_file, output_dir_qa, output_dir_text_gen)
