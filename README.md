# News Article Summarizer

A Streamlit application that generates concise summaries of news articles using a BART transformer model. This academic project demonstrates how a trained NLP model can be packaged behind a simple user interface.

## What this project demonstrates

- Transformer-based text summarization
- Hugging Face Transformers and PyTorch
- Preparing and tokenizing text for sequence-to-sequence models
- Publishing a trained model on Hugging Face Hub
- Loading a remote model for inference
- Building an interactive Streamlit interface

## How it works

```text
Article text
    ↓
BART tokenizer
    ↓
Fine-tuned BART model
    ↓
Beam-search generation
    ↓
Readable summary
```

The Streamlit application loads the model and tokenizer from Hugging Face, accepts article text and generates a summary with configurable generation constraints.

## Technology

- Python
- PyTorch
- Hugging Face Transformers
- BART
- Streamlit
- Jupyter Notebook
- CNN/DailyMail dataset

## Model

The model is available on Hugging Face:

[RamiBadleh/bart-news-summarizer](https://huggingface.co/RamiBadleh/bart-news-summarizer)

## Run locally

```bash
git clone https://github.com/AI-Rami/News-Article-Summarizer..git
cd News-Article-Summarizer.
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the model, so startup time depends on the machine and network connection.

## Repository contents

- `app.py`: Streamlit inference interface
- `news_summarizer_notebook.ipynb`: model-development workflow
- `requirements.txt`: Python dependencies

## Limitations

The current application truncates inputs to the model's maximum input length during inference. A production version should add robust long-document chunking, evaluation against reference summaries, caching, error handling and deployment monitoring.

## Context

Developed as an academic NLP project at Inland Norway University of Applied Sciences.

## Author

Rami — AI bachelor student with a professional background in chemistry, pharmaceutical quality control and regulated technical environments.
