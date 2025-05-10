from typing import List

class RoutingController:
    #Add semantic layer for query analysis
    def __init__(self): 
        self.manual_keywords = [
            "how to", "steps", "procedure", "workflow", "guide", "instruction", "document",
            "navigate", "manual", "usage", "configuration"
        ]
        self.jira_keywords = [
            "error", "issue", "build fails", "ticket", "jira", "deployment", "log", "solution",
            "failed", "bug", "problem", "not working", "fix"
        ]

    def route(self, query: str) -> List[str]:
        q = query.lower()
        manual = any(kw in q for kw in self.manual_keywords)
        jira = any(kw in q for kw in self.jira_keywords)

        if manual and not jira:
            return ["cap_manual_v3"]
        elif jira and not manual:
            return ["jira_tickets_hybrid"]
        else:
            return ["cap_manual_v3", "jira_tickets_hybrid"]
