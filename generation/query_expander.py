# generation/query_expander.py

import random
from query_ollama_llm import query_ollama
from generation.query_embedding_utils import get_embedding

class QueryExpander:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.model_name = model_name

    def expand_query_hyde(self, question):
        """
        Use HyDE (Hypothetical Document Embeddings) to generate an imaginary answer
        and re-embed it as an enriched query.
        """
        hyde_prompt = f"""You are a technical writer. Imagine you are writing a short, factual answer to the following question, based on a user manual.

        Question: {question}

        Write a concise answer (~5 lines max) as if it existed in the manual."""

        generated_answer = query_ollama(hyde_prompt, model=self.model_name)

        if not generated_answer:
            return get_embedding(question)

        # Embed the generated answer
        return get_embedding(generated_answer)

    def expand_query_variants(self, question):
        """
        Optionally, generate paraphrases if you want multiple expansions (to be done if we have time).
        """
        return [question]  

