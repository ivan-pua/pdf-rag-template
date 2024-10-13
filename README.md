# PDF-Bot: Building your first AI assistant 🤖

To build a chatbot that is able to perform tasks based on a PDF with English (e.g. Q&A, summarisation, content creation) 

**Update 13/10/2024**: For this particular HuggingFace model in `main.py`, only question and answering is supported. For a full range of chat features, please use the OpenAI model in `main_openai.py`

# Installation Steps
1. `conda create -n your_env_name python=3.11`
1. `conda activate your_env_name`
1. `pip install poetry`
1. `poetry init`
1. `poetry install --no-root`

# Run PDF-bot locally
1. `streamlit run website.py`
1. Go to http://localhost:8501 in your browser
