import streamlit as st
import os
import torch
import nltk
import urllib.request
from models.model_builder import ExtSummarizer
from newspaper import Article, ArticleException
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
    st.write("<h1 style='text-align : center'> Summarize and Ask </h1>",unsafe_allow_html=True)
    st.write("""
            <p style='text-align : justify'>
            Summarize and Ask was made by James Liang for the OSS submission of Sunday August 9th 2021
            </p>
            """,unsafe_allow_html=True)
    RAW_TEXT = "Raw Text"
    URL = "URL"
    AUTO_DETECT = "Auto Detect"
    placeholder = "Your text here"
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
        blockquoted_text = '\n'.join([f'> {line}' for line in limited_text.split('\n')])
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
    result_fp = "./results/summary.txt"
    model = st.session_state["summarizer"]
    st.write("## Summary")
    num_sentences = st.slider("Summary Length",3,8,step=1)
    summary = summarize(raw_fp, result_fp, model,max_length=num_sentences)
    summary = summarize(raw_fp, result_fp, model,max_length=num_sentences)
    bullet_points = '\n'.join(
        [f"<li> {sent} </li>" for sent in sent_tokenize(summary)])
    st.write(f"<ol> {bullet_points} </ol>", unsafe_allow_html=True)

    if len(summary) > 10:
        st.write("## Question Time!")
        placeholder = "Ask any question"
        question = st.text_input("", placeholder)
        if len(text.split()) > 400:
            context = summarize(raw_fp,result_fp,model,max_length=50)
        else:
            context = text
        if question == placeholder or question == "":
            return
        if len(question.split()) > 30:
            st.warning("Woah there, write less than 30 words for your question.")
            return 
        info = st.info("Hold on, we're answering your question!")
        context = ' '.join([word for word in context.split()][:480])
        answer = answer_question(question, context)
        info.empty()
        st.write(f"Answer : {answer}")


def write_raw_text(text):
    raw_fp = "./raw_data/input.txt"
    with open(raw_fp, "w") as f:
        f.write(text)
    return raw_fp


def answer_question(question, text):
    tokenizer, model = st.session_state["models"]
    inputs = tokenizer.encode_plus(
        question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    answer_start_scores, answer_end_scores = model(
        **inputs)[0], model(**inputs)[1]

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
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


def get_article(url):
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
