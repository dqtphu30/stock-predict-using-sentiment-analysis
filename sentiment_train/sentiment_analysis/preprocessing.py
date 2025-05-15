from underthesea import sent_tokenize, word_tokenize
from underthesea import ner
import nltk
import re
import string

with open("sentiment_analysis/stock-id.txt", "r", encoding="utf-8") as file:
    stock_id = file.read().split("\n")
stock_id = set(stock_id)

def Name_Entity_Recognize(text):
    doc = ner(text)
    temp = [i[0] for i in doc if i[1] in ['CH', 'Np', 'M', 'Nu', 'X', 'R']]
    ent = []
    for i in temp:
        i = str(i)
        i = i.lower()
        ent.append(i)
    return ent

def processtext(text):
    words = set(nltk.corpus.words.words())
    text = re.sub(r'\([A-Z\-0-9]{2,}\)?', '', text)
    #text = re.sub(r'\s\s+', ' ', text) ## Cắt bỏ khoảng trắng
    ent = Name_Entity_Recognize(text)
    text = word_tokenize(text)
    text = [w.lower() for w in text if w.lower() not in stock_id]
    text = [w for w in text if w not in ent]
    text = [w for w in text if len(w)!=1]
    text = [w for w in text if w not in words]
    #text = ' '.join(c for c in text if c <= '\uFFFF')
    text = [re.sub(r'[^\w\s]','', w) for w in text]## Loại bỏ dấu câu
    text = [re.sub(r'\b\d+\b', '', w) for w in text] ## Xóa số
    text = [re.sub(r'[\d]', '', w) for w in text] ## Xóa số
    text = [re.sub(r'\s\s+', '', w) for w in text] ## Cắt bỏ khoảng trắng
    text = [w for w in text if w != '']
    return text

def readData(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = text.split('\n')
    return sentences

def tokenizeWords(sentences):
    words = [processtext(sentence) for sentence in sentences]
    return words