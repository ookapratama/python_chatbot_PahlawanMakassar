from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """
    Memuat model Text Generation.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def inference_text_generation(question, context, model, tokenizer, max_length=50):
    """
    Menghasilkan jawaban dengan Text Generation.
    """
    prompt = f"{question} {context}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
