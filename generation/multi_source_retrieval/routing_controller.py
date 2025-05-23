from typing import List
from transformers import pipeline

class RoutingController:
    """
    Routes queries to the appropriate source(s) using both keyword heuristics
    and a semantic zero-shot classifier for more robust intent detection.
    """
    def __init__(self):
        # Keyword-based fallback for safety 
        self.manual_keywords = [
            "how to", "steps", "procedure", "workflow", "guide", "instruction", "document",
            "navigate", "manual", "usage", "configuration"
        ]
        self.jira_keywords = [
            "error", "issue", "build fails", "ticket", "jira", "deployment", "log", "solution",
            "failed", "bug", "problem", "not working", "fix"
        ]
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self.labels = ["manual", "jira"]
        self.threshold = 0.5

    def route(self, query: str) -> List[str]:
        q = query.strip()
        result = self.classifier(q, self.labels)
        scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        semantic_manual = scores.get('manual', 0) >= self.threshold
        semantic_jira = scores.get('jira', 0) >= self.threshold
        lower_q = q.lower()
        kw_manual = any(kw in lower_q for kw in self.manual_keywords)
        kw_jira = any(kw in lower_q for kw in self.jira_keywords)
        use_manual = semantic_manual or (kw_manual and not kw_jira)
        use_jira = semantic_jira or (kw_jira and not kw_manual)

        routes = []
        if use_manual:
            routes.append("cap_manual_v3")
        if use_jira:
            routes.append("jira_tickets_hybrid")

        if not routes:
            routes = ["cap_manual_v3", "jira_tickets_hybrid"]

        return routes
