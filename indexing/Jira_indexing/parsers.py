import xml.etree.ElementTree as ET
from typing import Dict, List
from datetime import datetime
from  indexing.utils  import clean_html

def parse_jira_xml(file_path: str) -> List[Dict]:
    tree = ET.parse(file_path)
    root = tree.getroot()
    tickets = []

    for item in root.findall(".//item"):
        key = item.findtext("key")
        title = item.findtext("title")
        if not key or not title:
            continue
        # Core metadata
        ticket = {
            "key": item.findtext("key"),
            "title": item.findtext("title"),
            "summary": item.findtext("summary"),
            "description": clean_html(item.findtext("description")),
            "status": item.findtext("status"),
            "resolution": item.findtext("resolution"),
            "priority": item.find("priority").get("id") if item.find("priority") else None,
            "created": item.findtext("created"),
            "updated": item.findtext("updated"),
            "labels": [label.text for label in item.findall("labels/label")],
            "reporter": item.find("reporter").get("username") if item.find("reporter") else None,
            "assignee": item.find("assignee").get("username") if item.find("assignee") else None,
        }

        # Parse comments
        ticket["comments"] = [
            {
                "author": comment.get("author"),
                "created": comment.get("created"),
                "text": clean_html(comment.text)
            }
            for comment in item.findall("comments/comment")
        ]

        # Parse attachments
        ticket["attachments"] = [
            {
                "id": attach.get("id"),
                "name": attach.get("name"),
                "url": f"https://eteamproject.internal.ericsson.com/secure/attachment/{attach.get('id')}/{attach.get('name')}"
            }
            for attach in item.findall("attachments/attachment")
        ]

        # Parse custom fields
        custom_fields = {
            field.findtext("customfieldname"): clean_html(
                field.findtext("customfieldvalues/customfieldvalue")
            )
            for field in item.findall("customfields/customfield")
        }
        ticket["custom_fields"] = custom_fields
        ticket["last_comment"] = custom_fields.get("Last Comment Body") or custom_fields.get("Last Comment", "")
        ticket["solution"] = custom_fields.get("Solution") or ""

        tickets.append(ticket)
    
    return tickets