import streamlit as st
import re
import string
import torch
import torch.nn.functional as F
from transformers import BertTokenizer


punctuations = re.sub(r"[!<_>#:)\.]", "", string.punctuation)

def punct2wspace(text):
    return re.sub(r"[{}]+".format(punctuations), " ", text)

def normalize_wspace(text):
    return re.sub(r"\s+", " ", text)

def casefolding(text):
    return text.lower()

def predict(text):
    text = punct2wspace(text)
    text = normalize_wspace(text)
    text = casefolding(text)

    model = torch.load("model.pkl", map_location='cpu')

    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

    for i, label in enumerate(labels):
        if label == 0:
            st.success(f'Label : NEGATIVE ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')        
        else:
            st.warning(f'Label : POSITIVE ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')

st.title('Hate Speech Recognition')
text = st.text_area('Tulis kalimat','...')
if st.button('Analyze'):
    if text is not None:
      with st.spinner('Analyzing the text …'):
        predict(text)
else:
     st.warning('Coba tulis Review')
     