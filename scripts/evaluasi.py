import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

nltk.download('wordnet')

def evaluate_text_generation(model_path, dataset_file, max_length=512, max_new_tokens=50):
    """
    Evaluasi model Text Generation menggunakan BLEU, ROUGE, dan METEOR.
    """
    # Load model dan tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Pastikan pad_token diatur jika belum ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    data = pd.read_csv(dataset_file)

    # Inisialisasi skor
    bleu_scores = []
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    meteor_scores = []

    # Inisialisasi ROUGE scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Iterasi melalui dataset
    for _, row in data.iterrows():
        prompt = row['prompt']
        target = row['target']

        # Hasilkan teks dari model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Tambahkan attention_mask
            max_new_tokens=max_new_tokens,  # Gunakan max_new_tokens untuk teks tambahan
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Tokenisasi reference dan candidate
        reference = target.split()  # Tokenisasi target
        candidate = generated_text.split()  # Tokenisasi teks yang dihasilkan

        # Hitung BLEU Score
        bleu_score = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu_score)

        # Hitung ROUGE Score
        rouge_score = rouge_scorer_instance.score(target, generated_text)
        rouge_scores["rouge-1"].append(rouge_score["rouge1"].fmeasure)
        rouge_scores["rouge-2"].append(rouge_score["rouge2"].fmeasure)
        rouge_scores["rouge-l"].append(rouge_score["rougeL"].fmeasure)

        # Hitung METEOR Score (gunakan daftar kata)
        meteor = meteor_score([reference], candidate)
        meteor_scores.append(meteor)

    # Rata-rata hasil evaluasi
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge_scores["rouge-1"]) / len(rouge_scores["rouge-1"])
    avg_rouge2 = sum(rouge_scores["rouge-2"]) / len(rouge_scores["rouge-2"])
    avg_rougel = sum(rouge_scores["rouge-l"]) / len(rouge_scores["rouge-l"])
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    # Tampilkan hasil
    print(f"Average BLEU Score: {avg_bleu:.2f}")
    print(f"Average ROUGE-1 Score: {avg_rouge1:.2f}")
    print(f"Average ROUGE-2 Score: {avg_rouge2:.2f}")
    print(f"Average ROUGE-L Score: {avg_rougel:.2f}")
    print(f"Average METEOR Score: {avg_meteor:.2f}")

# Contoh penggunaan
if __name__ == "__main__":
    model_path = "./tuning_gpt2_medium_indo/"
    dataset_file = "../dataset/text_generation_dataset.csv"
    evaluate_text_generation(model_path, dataset_file)
