import argparse
from train import train_qa, train_text_generation
from preproces import preprocess_data_for_qa, prepare_text_generation_dataset
from evaluasi import evaluate_text_generation
# from infrence import load_model, inference_text_generation

def main():
    parser = argparse.ArgumentParser(description="Pipeline Fine-Tuning QA → Text Generation")
    
    # Tambahkan opsi mode
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train_qa", "generate_text_gen_dataset", "train_text_gen", "evaluate_text_gen", "inference"],
        help="Mode yang akan dijalankan: train_qa, generate_text_gen_dataset, train_text_gen, evaluate_text_gen, inference."
    )
    
    # Tambahkan argumen umum
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B", help="Nama model pretrained.")
    parser.add_argument("--qa_model_path", type=str, help="Path ke model QA yang sudah dilatih.")
    parser.add_argument("--dataset_file", type=str, help="Path ke dataset.")
    parser.add_argument("--output_dir", type=str, help="Direktori output untuk menyimpan model.")
    parser.add_argument("--output_file", type=str, help="File output untuk dataset Text Generation.")
    parser.add_argument("--max_length", type=int, default=512, help="Panjang maksimal token.")
    parser.add_argument("--question", type=str, help="Pertanyaan untuk inferensi.")
    parser.add_argument("--context", type=str, help="Konteks untuk inferensi.")
    
    args = parser.parse_args()
    
    # Eksekusi berdasarkan mode
    if args.mode == "train_qa":
        print("Training model QA...")
        train_qa(args.model_name, args.dataset_file, args.output_dir, args.max_length)
    
    elif args.mode == "generate_text_gen_dataset":
        print("Membuat dataset Text Generation...")
        prepare_text_generation_dataset(
            args.dataset_file, args.qa_model_path, args.output_file, args.max_length
        )
    
    elif args.mode == "train_text_gen":
        print("Training model Text Generation...")
        train_text_generation(args.model_name, args.dataset_file, args.output_dir, args.max_length)
    
    elif args.mode == "evaluate_text_gen":
        print("Evaluasi model Text Generation...")
        evaluate_text_generation(args.qa_model_path, args.dataset_file, args.max_length)
    
    # elif args.mode == "inference":
    #     print("Inferensi dengan model Text Generation...")
    #     model, tokenizer = load_model(args.qa_model_path)
    #     answer = inference_text_generation(args.question, args.context, model, tokenizer, args.max_length)
    #     print(f"Jawaban: {answer}")
    
    else:
        print("Mode tidak dikenal!")

if __name__ == "__main__":
    main()