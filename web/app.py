from flask import Flask, render_template, request, jsonify
from controllers.infrence import text_generation_inference_only, find_prompt_from_text_gen_dataset, add_prompt_to_dataset

app = Flask(__name__)

TEXT_GEN_DATASET_FILE = "../dataset/text_generation_dataset.csv"
GEN_MODEL_PATH = "../scripts/tuning_gpt2_medium_indo/"


@app.get("/")
def main():
	return render_template('index.html')

@app.route('/handlePrompt', methods=['POST'])
def handlePrompt() :
    prompt = request.form.get('prompt', '').strip()
    target = request.form.get('target', '').strip() 
    

    if not prompt:
        return jsonify({'error': 'Prompt tidak boleh kosong'}), 400

    if not target:
        # Cari target di dataset
        target = find_prompt_from_text_gen_dataset(prompt, TEXT_GEN_DATASET_FILE)
        if target:
            print("generate text from dataset")
            return jsonify({'generated_text': target})
        else:
            # Jika tidak ditemukan, fallback ke text generation
            # try:
            #     print("generate text from model")
            #     generated_text = text_generation_inference_only(
            #         prompt=prompt,
            #         dataset_file=TEXT_GEN_DATASET_FILE,
            #         model_path=GEN_MODEL_PATH,
            #         max_new_tokens=50
            #     )
            #     return jsonify({'generated_text': generated_text})
            # except Exception as e:
            #     return jsonify({'error': str(e)}), 500
            print("Prompt tidak ditemukan di dataset.")
            return jsonify({'error': 'Prompt tidak ditemukan. Silakan tambahkan target baru.'}), 404
    else:
        # Tambahkan prompt dan target baru ke dataset
        success = add_prompt_to_dataset(prompt, target, TEXT_GEN_DATASET_FILE)
        if success:
            return jsonify({'message': 'Prompt dan target baru berhasil ditambahkan ke dataset.'})
        else:
            return jsonify({'error': 'Gagal menambahkan prompt dan target ke dataset.'}), 500

if __name__ == "__main__" :
    app.run(debug=True)