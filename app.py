import streamlit as st
import os
import torch
import nltk
import urllib.request
from models.model_builder import ExtSummarizer
from newspaper import Article, ArticleException
from ext_sum import summarize
from transformers import pipeline
import transformers
import torch
from typing import Any
from nltk.tokenize import sent_tokenize
from tokenizers import Tokenizer

RESULT_FP = "./results/summary.txt"
RAW_FP = "./raw_data/input.txt"


@st.cache(suppress_st_warning=True, show_spinner=False)
def load_model(model_type):
    checkpoint = torch.load(
        f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtSummarizer(
        device="cpu", checkpoint=checkpoint, bert_type=model_type)
    return model

@st.cache(suppress_st_warning=True,show_spinner=False,hash_funcs={Tokenizer : id})
def load_qa():
    return pipeline("question-answering")

@st.cache(suppress_st_warning=True, show_spinner=False)
def summarize_cached(input_text, model, max_length):
    return summarize(RAW_FP, RESULT_FP, model, max_length=max_length)


def main():
    st.write("<h1 style='text-align : center'> Summarize and Ask </h1>",
             unsafe_allow_html=True)
    st.write("""
            <p style='text-align : justify'>
            Summarize and Ask was made by James Liang for the OSS submission of Sunday August 9th 2021
            </p>
            """, unsafe_allow_html=True)
    RAW_TEXT = "Raw Text"
    URL = "URL"
    AUTO_DETECT = "Auto Detect"
    placeholder = "Your text here"
    model = load_model("mobilebert")
    raw_input_type = st.radio("Input format", [AUTO_DETECT, RAW_TEXT, URL])
    if raw_input_type == AUTO_DETECT:
        RAW_TEXT_PLACEHOLDER = "Paste a URL or a text"
        raw_text = st.text_area("", RAW_TEXT_PLACEHOLDER)
        if raw_text == RAW_TEXT_PLACEHOLDER or raw_text == "":
            return
        if len(raw_text.split()) == 1 and raw_text.startswith("http"):
            input_type = URL
            url = raw_text
        else:
            input_type = RAW_TEXT
            text = raw_text

    input_type = raw_input_type if raw_input_type != AUTO_DETECT else input_type
    if raw_input_type == RAW_TEXT:
        text = st.text_area("", placeholder)
    elif raw_input_type == URL:
        url = st.text_input("", "URL here")

    if input_type == RAW_TEXT:
        if text == placeholder or text == "" or len(text.split()) < 10:
            st.info("Type more than 10 words")
            return
    elif input_type == URL:
        valid_url = True
        try:
            text = get_article(url)
        except ArticleException:
            valid_url = False
        if not valid_url:
            st.info("Enter a valid url")
            return
        if len(text.split()) < 10:
            st.info("Enter a URL that contains enough text")
            return
        limited_text = " ".join(sent_tokenize(text)[:5])
        blockquoted_text = '\n'.join(
            [f'> {line}' for line in limited_text.split('\n')])
        st.write("## URL Preview")
        st.write(blockquoted_text)
        st.write(f"Continue reading [here]({url})")
        if abs(len(text) - len(limited_text)) < 10:
            st.warning("""
                    We could not find more words other than the preview shown above. 
                    This occurs when content is locked for the public (news subscription / Medium).
                    Please be aware of it if you proceed.
                    Alternatively, you can paste the full text instead of the URL.
                    """)
    raw_fp = write_raw_text(text)
    st.write("## Summary")
    num_sentences = st.slider("Summary Length", 3, 8, step=1)
    summary = summarize_cached(text, model, max_length=num_sentences)
    bullet_points = '\n'.join(
        [f"<li> {sent} </li>" for sent in sent_tokenize(summary)])
    st.write(f"<ol> {bullet_points} </ol>", unsafe_allow_html=True)

    if len(summary) > 10:
        st.write("## Question Time!")
        placeholder = "Ask any question"
        question = st.text_input("", placeholder)
        if question == placeholder or question == "":
            return
        if len(question.split()) > 30:
            st.warning(
                "Woah there, write less than 30 words for your question.")
            return
        info = st.info("Hold on, we're answering your question!")
        if len(text.split()) > 400:
            context = summarize_cached(text, model, max_length=20)
        else:
            context = text
        context = ' '.join([word for word in context.split()][:480])
        qa = load_qa()
        answer, score = answer_question(question,context,qa)
        info.empty()
        if score < 0.05:
            st.warning(f"The model didn't know what to reply. Make sure you ask a valid question.")
            st.info("This is what the model outputted, read at your own risk")
            st.write(f"Answer : {answer}")

        else:
            st.write(f"Answer : {answer}")

@st.cache(suppress_st_warning=True,show_spinner=False,hash_funcs={Tokenizer : id})
def answer_question(question,context,qa):
    output = qa(question=question,context=context)
    return output['answer'], output['score']

def write_raw_text(text):
    with open(RAW_FP, "w") as f:
        f.write(text)
    return RAW_FP


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


def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


if __name__ == "__main__":
    st.set_page_config(page_title="Summarize and Ask",page_icon=":book")
    if not os.path.exists("checkpoints/mobilebert_ext.pt"):
        download_model()
    main()
