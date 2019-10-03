# https://github.com/prateekjoshi565/textrank_text_summarization/blob/master/TestRank_Text_Summarization.ipynb
# https://blog.csdn.net/m0_37700507/article/details/84726463
# https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/85180300

import numpy as np
import pandas as pd
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from nltk.tokenize import sent_tokenize

nltk.download('punkt') # one time execution
import re

pd.set_option('display.max_columns', 1000)
df = pd.read_csv('tennis_articles_v4.csv')
# print(df.head())

# split the the text in the articles into sentences
sentences = []
for s in df['article_text']:
    sentences.append(sent_tokenize(s))


sentences = [y for x in sentences for y in x]


# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


nltk.download('stopwords')# one time execution


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove stopwords from the sentences
# 对文本数据做一些基本的文本清理以尽可能避免文本数据的噪音对摘要提取的影响。
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# Extract word vectors
# 提取单词嵌入或单词向量
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

# 句子的向量表示
# 句子的向量表示是用句子中各个单词的向量的平均数表示
sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)




# 下一步是找出句子之间的相似之处。
# 我们将使用余弦相似度来寻找一对句子之间的相似度。
# 让我们为这个任务创建一个空的相似度矩阵，并用句子的余弦相似度填充它。


# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


# 在这里我们将相似矩阵sim_mat转换为图形。
# 图中的节点表示句子，边表示句子之间的相似度得分。
# 在这个图中，我们将使用PageRank算法得到句子的排名。
import networkx as nx
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Specify number of sentences to form the summary
# 生成的句子数
sn = 10

# Generate summary
# 根据句子所得的分，生成最重要的前n句作为文章的自动摘要。
for i in range(sn):
    print(ranked_sentences[i][1])




