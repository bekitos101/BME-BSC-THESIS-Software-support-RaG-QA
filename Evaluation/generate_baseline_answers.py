import json
import os
import time
from PyPDF2 import PdfReader
from openai import OpenAI  
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/user_manual_cleaned.pdf"
EVAL_JSON_PATH = "manual_questions.json"
OUTPUT_JSON_PATH = "evaluation_with_baseline_gpt.json"
DELAY_BETWEEN_CALLS = 60

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())


def build_prompt(question, full_context):
    return f"""You are a helpful assistant for answering documentation questions based on a full user manual.

Answer the following question using the provided context. Be accurate, concise, and do not hallucinate. If unsure, say you don't know.

Context:
{full_context}

Question: {question}

Answer:"""

def query_gpt(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You answer documentation-related queries with accuracy and conciseness."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def run_baseline_gpt():
    full_manual_text = extract_text_from_pdf(PDF_PATH)
    with open(EVAL_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        question = entry["question"]
        print(f"[{i+1}/{len(data)}] Querying baseline for: {question}")

        prompt = build_prompt(question, user_manual)
        answer = query_gpt(prompt)
        print(answer)

        entry["baseline_answer"] = answer
        # Required for RAGAS
        entry["baseline_retrieved_chunks"] = [user_manual]  

        time.sleep(DELAY_BETWEEN_CALLS)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nGPT baseline generation complete. Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    run_baseline_gpt()
