import streamlit as st
import os
import torch
import nltk
import urllib.request
from models.model_builder import ExtSummarizer
from newspaper import Article
from ext_sum import summarize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import Any
from nltk.tokenize import sent_tokenize


@st.cache(suppress_st_warning=True)
def load_model(model_type):
    checkpoint = torch.load(
        f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtSummarizer(
        device="cpu", checkpoint=checkpoint, bert_type=model_type)
    return model


def load_qna():
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    return tokenizer, model

def main():
    st.markdown("<h1 style='text-align: center;'>Summarize and Ask ✏️</h1>",
                unsafe_allow_html=True)

    # Download model

    # Input
    input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
    st.markdown("<h3 style='text-align: center;'>Input</h3>",
                unsafe_allow_html=True)

    if input_type == "Raw Text":
        with open("raw_data/input.txt") as f:
            sample_text = f.read()
        text = st.text_area("", sample_text, 200)
    else:
        url = st.text_input(
            "", "https://www.cnn.com/2020/05/29/tech/facebook-violence-trump/index.html")
        st.markdown(f"[*Read Original News*]({url})")
        text = crawl_url(url)

    input_fp = "raw_data/input.txt"
    with open(input_fp, 'w') as file:
        file.write(text)

    # Summarize
    model = st.session_state["summarizer"]
    sum_level = st.radio("Output Length: ", ["Short", "Medium"])
    max_length = 3 if sum_level == "Short" else 5
    result_fp = 'results/summary.txt'
    summary = summarize(input_fp, result_fp, model, max_length=max_length)
    points = sent_tokenize(summary)
    li_points = '\n'.join([f"<li>{point}</li>" for point in points])
    
    st.markdown("<h3 style='text-align: center;'>Summary</h3>",
                unsafe_allow_html=True)
    st.markdown(f"<ol align='justify'>{li_points}</ol>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align : center'>Ask A Question </h3>",
                unsafe_allow_html=True)
    placeholder = "Ask any question..."
    question = st.text_input("",placeholder)
    if question != "" and question != placeholder:
        answer = answer_question(question, text)
        st.markdown(
            f"<p style='text-align : justify'> {answer}</p>", unsafe_allow_html=True)


def answer_question(question, text):
    tokenizer, model = st.session_state["models"]
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
 
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]
 
    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def download_model():
    nltk.download('popular')
    url = 'https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading checkpoint...")
        progress_bar = st.progress(0)
        with open('checkpoints/mobilebert_ext.pt', 'wb') as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
                                            (counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


if __name__ == "__main__":
    if 'models' not in st.session_state:
        st.session_state.models = load_qna()
    if 'summarizer' not in st.session_state:
        if not os.path.exists('checkpoints/mobilebert_ext.pt'):
            download_model()
        st.session_state.summarizer = load_model("mobilebert")
    main()
