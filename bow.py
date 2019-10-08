import pandas as pd
import re
from gensim.models import KeyedVectors

w2v = KeyedVectors.load_word2vec_format("C:\\HOCTAP\\python\\NLP\\vi.vec")
vocab = w2v.wv.vocab 

filename = 'C:\\Users\\16521\\Downloads\\project\\Data-Sentiment-Analysis\\train.csv'
df = pd.read_csv(filename)
bow = []
for i in df['text']:
    x = i.lower().strip().replace('\n',"")
    x = re.sub(r'([a-z])\1+', lambda m: m.group(1), x, flags=re.IGNORECASE)
    x = re.findall(r'[a-z]\w+', x)
    x = [word for word in x if word.isalpha()]
    for j in x:
        if j not in vocab:
            x.remove(j)
    bow +=x
words = set(bow)
worddictA = dict.fromkeys(words, 0)
for word in bow:
    worddictA[word] +=1
# print(len(worddictA))
# print(list(worddictA.values()))
x = list(worddictA.keys())
y = list(worddictA.values())
dic = {'words': x, 'bow': y}
df = pd.DataFrame(dic)
csv = df.to_csv('bow.csv', encoding="utf-8")
print(df.head())

