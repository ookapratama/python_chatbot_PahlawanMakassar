import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForCausalLM, AutoTokenizer
from preproces import preprocess_data
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate_models(qa_model_path, text_gen_model_path, dataset_file, qa_model_name, text_gen_model_name, max_length=512):
    qa_tokenized_data, text_gen_tokenized_data = preprocess_data(dataset_file, qa_model_name, text_gen_model_name, max_length)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path)
    text_gen_model = AutoModelForCausalLM.from_pretrained(text_gen_model_path)

    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    text_gen_tokenizer = AutoTokenizer.from_pretrained(text_gen_model_name)

    correct = 0
    total = len(qa_tokenized_data)
    pred_starts, true_starts, pred_ends, true_ends, distances = [], [], [], [], []

    for data in qa_tokenized_data:
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in data.items() if k not in ['start_positions', 'end_positions']}
        outputs = qa_model(**inputs)
        start_pred = torch.argmax(outputs.start_logits).item()
        end_pred = torch.argmax(outputs.end_logits).item()
        start_true, end_true = data['start_positions'], data['end_positions']
        pred_starts.append(start_pred); true_starts.append(start_true)
        pred_ends.append(end_pred); true_ends.append(end_true)
        distances.append(abs(start_pred - start_true) + abs(end_pred - end_true))
        if start_pred == start_true and end_pred == end_true:
            correct += 1

    accuracy = correct / total
    print(f"Akurasi QA: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    qa_model_path = "./fine_tuned_model_qa/"
    text_gen_model_path = "./fine_tuned_model_text_gen/"
    dataset_file = "../dataset/enhanced_qa_dataset.csv"
    qa_model_name = "malaputri/indobert-squad-id"
    text_gen_model_name = "TurkuNLP/gpt3-finnish-small"
    
    evaluate_models(qa_model_path, text_gen_model_path, dataset_file, qa_model_name, text_gen_model_name)
