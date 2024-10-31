import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from preproces import preprocess_data
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def evaluate_model(model_path, dataset_file, model_name, max_length=512):
    """
    Mengevaluasi model QA dengan metrik tambahan.

    Args:
    - model_path: Path model yang sudah dilatih.
    - dataset_file: Dataset CSV untuk evaluasi.
    - model_name: Nama model yang digunakan.
    - max_length: Panjang maksimal token.
    """
    # Load dataset yang sudah dipreproses
    tokenized_data = preprocess_data(dataset_file, model_name, max_length)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    correct = 0
    total = len(tokenized_data)
    pred_starts = []
    true_starts = []
    pred_ends = []
    true_ends = []
    distances = []

    for data in tokenized_data:
        # Siapkan input tanpa 'start_positions' dan 'end_positions'
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in data.items() if
                  k not in ['start_positions', 'end_positions']}

        # Dapatkan prediksi model
        outputs = model(**inputs)
        start_pred = torch.argmax(outputs.start_logits).item()
        end_pred = torch.argmax(outputs.end_logits).item()

        # Ambil nilai start dan end yang benar dari dataset
        start_true = data['start_positions']
        end_true = data['end_positions']

        # Simpan nilai untuk perhitungan precision, recall, dan f1
        pred_starts.append(start_pred)
        true_starts.append(start_true)
        pred_ends.append(end_pred)
        true_ends.append(end_true)

        # Hitung jarak prediksi dari posisi sebenarnya
        distances.append(abs(start_pred - start_true) + abs(end_pred - end_true))

        # Hitung akurasi per token
        if start_pred == start_true and end_pred == end_true:
            correct += 1

    # Akurasi
    accuracy = correct / total

    # Precision, Recall, F1 Score
    precision = precision_score(true_starts + true_ends, pred_starts + pred_ends, average='macro')
    recall = recall_score(true_starts + true_ends, pred_starts + pred_ends, average='macro')
    f1 = f1_score(true_starts + true_ends, pred_starts + pred_ends, average='macro')

    # Jarak rata-rata
    avg_distance = np.mean(distances)

    # Tampilkan semua skor
    print(f"Akurasi: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Average Distance: {avg_distance:.2f} tokens")


if __name__ == "__main__":
    model_path = "./fine_tuned_model/"  # Path model yang sudah dilatih
    dataset_file = "../dataset/qa_dataset.csv"  # Path dataset
    # model_name = "mrm8488/bert-small-finetuned-squadv2"  # Model yang digunakan
    # model_name = "Wikidepia/indobert-lite-squad"  
    model_name = "malaputri/indobert-squad-id"  

    evaluate_model(model_path, dataset_file, model_name)
