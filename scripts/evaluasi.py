from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def evaluate_text_generation(model_path, dataset_file, max_length=512):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = pd.read_csv(dataset_file)

    scores = []
    for _, row in data.iterrows():
        prompt = row['prompt']
        target = row['target']

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=max_length)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        reference = target.split()
        candidate = generated_text.split()
        score = sentence_bleu([reference], candidate)
        scores.append(score)

    print(f"Average BLEU Score: {sum(scores) / len(scores):.2f}")
