import requests

def query_ollama(prompt, model="deepseek-r1:1.5b", stream=False):
    """
    Send a prompt to the local Ollama server and return the model response.

    Args:
        prompt (str): The text prompt to send.
        model (str): The name of the local model served by Ollama.
        stream (bool): Whether to stream the output (not used here).

    Returns:
        str or None: The model's response or None if an error occurs.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        print("Ollama call failed:", e)
        if response is not None:
            print(response.text)
        return None
