📰 News Article Summarizer

This project is a web-based application that summarizes long news articles using state-of-the-art Natural Language Processing (NLP) techniques. It leverages the BART transformer model (facebook/bart-large-cnn) and the CNN/DailyMail dataset to automatically generate concise, readable summaries.

The app was developed as part of a school project and showcases how modern NLP models can be applied to real-world text processing tasks. It includes a custom preprocessing pipeline that handles articles exceeding the model’s token limit by applying sentence-based chunking.

FEATURES

Summarizes long news articles automatically
Uses the BART transformer model for text generation
Implements sentence-based chunking for inputs over 1024 tokens
Pretrained model fine-tuned on CNN/DailyMail dataset
Structured, modular Python codebase for easy extension or UI integration
TECH STACK

Python
Hugging Face Transformers
PyTorch
CNN/DailyMail Dataset
Jupyter Notebook (for training and testing)
Streamlit for deployment
PROJECT STRUCTURE

News-Article-Summarizer/ ├── app.py ├── trained_bart_summarization_model/ ├── news_summarizer_notebook.ipynb ├── requirements.txt ├── README.txt

HOW IT WORKS

Load 500 articles from CNN/DailyMail
Preprocess: tokenize, chunk long inputs, and pad sequences
Feed into BART model
Decode output into summaries
LICENSE

This project is licensed under the MIT License.

ACKNOWLEDGMENTS

Hugging Face Transformers
CNN/DailyMail Dataset
Developed as a school project
