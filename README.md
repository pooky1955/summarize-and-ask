# Summarize And Ask
Summarizer + QnA using StreamLit and MobileBertSum

Streamlit Starter code and MobileBERTSum Code provided by : https://github.com/chriskhanhtran/bert-extractive-summarization/

It uses the transformers BERT to answer questions and MobileBertSum to perform extractive summarization.

Made by James Liang for a OSS Sunday Challenge (Sunday August 8th 2021)

## Demo
1. Here is a demo : https://youtu.be/N7ZtIir_3mM

## Setting Up
1. Git clone the repository using `git clone https://github.com/pooky1955/summarize-and-ask.git`
2. Once inside the project directory, (assuming you have a working Python > 3.8 environment), run `pip install -r requirements.txt`
3. Download the pretrained MobileBertSum with `wget -O "checkpoints/mobilebert_ext.pt" "https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"`
4. Run the webapp using `streamlit run app.py`

```sh
git clone https://github.com/pooky1955/summarize-and-ask.git
cd summarize-and-ask
pip install -r requirements.txt
wget -O "checkpoints/mobilebert_ext.pt" "https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
streamlit run app.py
```
