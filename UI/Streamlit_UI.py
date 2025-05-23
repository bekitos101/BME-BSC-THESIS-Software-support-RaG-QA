import streamlit as st
import sys
import os
import json
import re
import base64

from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add generation module 
generation_path = os.path.abspath(os.path.join(current_dir, "..", "generation"))
if generation_path not in sys.path:
    sys.path.insert(0, generation_path)

# Add multi source retriever module
multi_path = os.path.abspath(os.path.join(current_dir, "..", "multi_source_retriever"))
if multi_path not in sys.path:
    sys.path.insert(0, multi_path)
    
#Add logo path
logo_path = os.path.join(current_dir, "static", "logo.png")
logo = Image.open(logo_path)
    

from chat_assistant import ChatAssistant

# UI CONFIG 
st.set_page_config(page_title="CAP Assistant", layout="wide")

# Load image and convert to base64
logo_path = os.path.join(current_dir, "static", "logo.webp")
with open(logo_path, "rb") as f:
    logo_data = f.read()
logo_base64 = base64.b64encode(logo_data).decode()

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0;">
    <img src="data:image/webp;base64,{logo_base64}" width="45" style="margin-top: 2px;">
    <h1 style="margin: 0; padding: 0;">  CAP Assistant</h1>
</div>
<p style="margin-top: 0.25rem;">Ask a question and choose a source. The assistant will retrieve context and generate an answer with references.</p>
""", unsafe_allow_html=True)

# SESSION STATE 
if "response" not in st.session_state:
    st.session_state.response = None

# INPUT FORM 
with st.form("qa_form"):
    question = st.text_area("Your question:")
    source = st.radio("Select source:", ["User Manual", "Support Tickets", "Multi Source"], horizontal=True)
    model_choice = st.selectbox("Choose LLM model:", ["Ollama deepSeek 1.5B", "GPT 3.5 Turbo"])
    model_name = ChatAssistant.LOCAL_MODEL if model_choice.startswith("Ollama") else ChatAssistant.CLOUD_MODEL
    submit = st.form_submit_button("Ask")

# PROCESS QUESTION 
if submit and question:
    if source == "User Manual":
        source_key = "cap_manual_v3"
    elif source == "Support Tickets":
        source_key = "jira_tickets_hybrid"
    else:
        source_key = "multi"

    assistant = ChatAssistant(model_name=model_name)
    try:
        result = assistant.ask(question, source=source_key, return_chunks=True)
        st.session_state.response = result
    except Exception as e:
        st.session_state.response = None
        st.error(f" Error: {e}")

#  DISPLAY RESPONSE 
if st.session_state.response:
    answer = st.session_state.response["answer"]
    references = st.session_state.response["references"]
    chunks = st.session_state.response["chunks"]

    # Extract and hide <think> section
    if answer.startswith("<think>"):
        end_of_think = answer.find("</think>") + len("</think>")
        think_section = answer[:end_of_think]
        main_answer = answer[end_of_think:].strip()
    else:
        think_section = ""
        main_answer = answer

    with st.expander("🔍 Internal Reasoning (Hidden)"):
        st.markdown(f"```text\n{think_section}\n```")

    st.markdown(main_answer)
    st.markdown("---")
    st.markdown("### 📚 Retrieved References")

    for ref, chunk in zip(references, chunks):
        key_match = re.search(r"\[Ticket (\w+-\d+)", chunk)
        title_match = re.search(r"Title: (.*?) \|", chunk)
        link_match = re.search(r"Link: (https?://[^\s]+)", chunk)

        ticket_key = key_match.group(1) if key_match else ref
        title = title_match.group(1) if title_match else "View Ticket"
        link = link_match.group(1) if link_match else "#"

        if "Page" in ref:
            display_title = f"📘 {ref}"
        else:
            display_title = f"🛠️ [{ticket_key} — {title}]({link})"


        formatted_chunk = (
            chunk
            .replace("Description:", "**📄 Description:**")
            .replace("Last Comment:", "**💬 Last Comment:**")
            .replace("Solution:", "**🧩 Solution:**")
            .replace("Status:", "**📌 Status:**")
            .replace("Comments:", "**💬 Comments:**")
            .replace("Labels:", "**🏷️ Labels:**")
            .replace("Link:", "**🔗 Link:**")
        )

        with st.container():
            with st.expander(display_title, expanded=False):
                st.markdown(
                    f"<div style='word-wrap: break-word; overflow-wrap: break-word; font-size: 0.92rem;'>{formatted_chunk}</div>",
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown("<div style='text-align: center;'>Built with 💙 for BME Bsc Thesis Project</div>", unsafe_allow_html=True)
