import sys
import os

indexing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Indexing"))
if indexing_path not in sys.path:
    sys.path.insert(0, indexing_path)

from multi_source_retrieval.multi_source_retriever import MultiSourceRetriever
from multi_source_retrieval.routing_controller import RoutingController
from qdrant_client import QdrantClient
from query_ollama_llm import query_ollama
from query_openai import query_openai

from generation.jira_retriever import JiraHybridRetriever
from generation.doc_retriever import SoftHybridRetriever

class ChatAssistant:
    LOCAL_MODEL = "deepseek-r1:1.5b"
    CLOUD_MODEL = "gpt-3.5-turbo"
    def __init__(self, model_name=CLOUD_MODEL, sources=None):
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.model_name = model_name
        self.sources = sources or ["cap_manual_v3", "jira_tickets_hybrid"]
        self.retrievers = {
            "cap_manual_v3": SoftHybridRetriever(collection_name="cap_manual_v3", host=self.qdrant_host, port=self.qdrant_port),
            "jira_tickets_hybrid": JiraHybridRetriever(collection_name="jira_tickets_hybrid", host=self.qdrant_host, port=self.qdrant_port, model_name=self.model_name),
            "multi": MultiSourceRetriever(model_name=self.model_name),

        }

    #Helper method
    def is_step_question(self, question):
        keywords = ["steps", "procedure", "how do i", "how to", "workflow", "process"]
        question = question.lower()
        return any(keyword in question for keyword in keywords)

    def build_prompt_for_manual(self, question, context):
        if self.is_step_question(question):
            return f"""You are a CPI Automation Portal documentation assistant.
            Use the following official manual excerpts to answer the question accurately.
            If the context contains steps or procedures, present them clearly and concisely as a step-by-step guide.
            Do not invent steps or instructions that are not present in the context.
        Context:
        {context}

        Question:
        {question}

        Answer:"""
        else:
            return f"""YYou are a CPI Automation Portal documentation assistant.
        Use the following official manual excerpts to answer the question accurately.
        Base your answer strictly on the provided context.
        Do not guess, do not invent information, and do not assume steps unless explicitly described.
        If the context is insufficient to fully answer the question, say so politely.
        .

            Context:
            {context}

            Question:
            {question}

            Answer:"""

    def build_prompt_for_tickets(self, question, context):
        if self.is_step_question(question):
            return f"""You are a helpful assistant for Jira support tickets.

                    Use the provided ticket context to answer the question.
                    Organize your answer into clear steps if possible.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:"""
        else:
            return f"""You are a helpful assistant for Jira support tickets.

                    Use the provided ticket context to summarize, explain, and answer naturally.
                    You can reason and synthesize ideas.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:"""
    def build_prompt_for_multi(self, question, context):
        return f"""You are a support assistant for CPI Automation Portal combining both user documentation and Jira ticket insights.
        Use the mixed context from both the manual and support tickets to give the best possible answer.
        Do not fabricate anything. If steps are needed, give them clearly.

        Context:
        {context}

        Question:
        {question}

        Answer:"""

    def ask(self, question, source="cap_manual_v3", return_chunks=False):
        retriever = self.retrievers.get(source)
        if not retriever:
            raise ValueError(f"Invalid source {source}. Must be one of {list(self.retrievers.keys())}.")
        context, references, chunks = retriever.retrieve(question)
        if source == "cap_manual_v3":
            prompt = self.build_prompt_for_manual(question, context)
        elif source == "jira_tickets_hybrid":
            prompt = self.build_prompt_for_tickets(question, context)
        elif source == "multi":
            routing = RoutingController()
            if "cap_manual_v3" in routing.route(question):
                prompt = self.build_prompt_for_manual(question, context)
            else:
                prompt = self.build_prompt_for_tickets(question, context)
        else:
            raise ValueError(f"Invalid source {source}")
        if self.model_name.startswith("gpt"):
            answer = query_openai(prompt, model=self.model_name)
        else:
            answer = query_ollama(prompt, model=self.model_name)
            
        reference_block = "\n\nReferences used:\n" + "\n".join(f"- {r}" for r in references)
        final_answer = (answer + reference_block) if answer else "No response generated."

        if return_chunks:
            return {
                "answer": final_answer,
                "references": references,
                "chunks": chunks
            }
        else:
            return final_answer