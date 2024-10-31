from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd

# Load the dataset
file_path = '../dataset/sultanHasanuddin.csv'
df = pd.read_csv(file_path)

# Model setup: mBERT for Indonesian language compatibility
model_name = "bert-base-multilingual-cased"  # Can also use IndoBERT if desired
# model_name = "indobenchmark/indobert-large-p2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load the context and question-answer data from the dataset
contexts = df['context'].tolist()
questions = df['question'].tolist()
answers = df['answer'].tolist()


def ask_question(question, contexts, detail_level="short"):
    """
    Function to ask questions with detail control.
    :param question: User's question
    :param contexts: List of context paragraphs from the dataset
    :param detail_level: Desired answer detail level ("short" or "detailed")
    :return: Relevant answer from the best-matching context
    """
    best_answer = ""
    max_score = 0
    best_context = ""

    # Find the best answer from all contexts
    for context in contexts:
        result = qa_pipeline(question=question, context=context)
        if result['score'] > max_score:
            max_score = result['score']
            best_answer = result['answer']
            best_context = context

    # If a detailed answer is requested, append the best context information
    if detail_level == "detailed":
        best_answer = f"{best_answer}. Informasi tambahan: {best_context}"
    print('score : ', result)
    return best_answer


# Chatbot interaction loop
print("Mulai bertanya tentang Sultan Hasanuddin! (Ketik 'q' untuk keluar)")

while True:
    user_question = input("Pertanyaan: ")
    if user_question.lower() == "q":
        break

    detail = input("Ingin jawaban singkat atau rinci? (short/detailed): ").strip().lower()
    answer = ask_question(user_question, contexts, detail_level=detail)

    print(f"Jawaban: {answer}\n")
