from preproces import prepare_text_generation_dataset

if __name__ == "__main__":
    qa_model_path = "./fine_tuned_model_qa/"
    dataset_file = "../dataset/enhanced_qa_dataset.csv"
    output_file = "../dataset/text_generation_dataset.csv"
    max_length = 512

    prepare_text_generation_dataset(dataset_file, qa_model_path, output_file, max_length)
