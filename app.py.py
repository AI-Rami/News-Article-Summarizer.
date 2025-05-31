# -*- coding: utf-8 -*-
"""
Created on Sun May 11 22:25:59 2025
@author: ramib
"""

import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import os

# to get the current directory of the app.py script
current_dir = os.path.dirname(os.path.abspath(__file__))

# this wil define the model path 
model_path = os.path.join(current_dir, 'bart_summarization_model')

# loading  the model and the tokenizer
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

#  title
st.title(" News Article Summarizer")

# input (article)
article_text = st.text_area("Enter the news article text here:")

# generate the summary
if st.button("Summarize"):
    if article_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        
        inputs = tokenizer.encode(article_text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, num_beams=4, max_length=150, min_length=30, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # show the summary
        st.subheader("Summary:")
        st.write(summary)
