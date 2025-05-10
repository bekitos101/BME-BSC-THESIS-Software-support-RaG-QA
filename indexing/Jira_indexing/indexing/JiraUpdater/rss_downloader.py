#rss_downloader.py
import requests
def fetch_jira_rss(xml_url: str, session_id: str, save_path: str = "SearchRequest_v2.xml"):
    """
    Downloads the latest Jira XML export using a browser session ID.
    
    Args:
        xml_url (str): The full Jira XML export URL (with JQL query).
        session_id (str): The value of your browser's JSESSIONID cookie.
        save_path (str): Where to save the downloaded XML file.
    """
    headers = {
        "Cookie": f"JSESSIONID={session_id}"
    }

    print(f"Fetching XML from: {xml_url}")
    response = requests.get(xml_url, headers=headers)

    if response.status_code == 200:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"XML saved to {save_path}")
    else:
        print(f"Failed to fetch XML: HTTP {response.status_code}")
