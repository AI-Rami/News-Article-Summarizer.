import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load model and tokenizer from Hugging Face Hub
model_name = "RamiBadleh/bart-news-summarizer"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Streamlit app layout
st.set_page_config(page_title="News Article Summarizer")
st.title("📰 News Article Summarizer")
st.write("Enter a news article below and get a concise summary using a fine-tuned BART model.")

# Text input
article_text = st.text_area("Paste the article here:", height=300)

# Summarization trigger
if st.button("Summarize"):
    if article_text.strip() == "":
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        # Tokenize and summarize
        inputs = tokenizer(article_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=150,
            min_length=30,
            early_stopping=True
        )
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

        # Output
        st.subheader("📝 Summary:")
        st.success(summary)
