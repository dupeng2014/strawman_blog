# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#nltk
import nltk

#stop-words
from nltk.corpus import stopwords
stop_words=set(nltk.corpus.stopwords.words('english'))

# tokenizing
from nltk import word_tokenize,sent_tokenize

#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model





# 创建文档的样本库，即文本
sample_text_1="bitty bought a bit of butter"
sample_text_2="but the bit of butter was a bit bitter"
sample_text_3="so she bought some better butter to make the bitter butter better"

corp=[sample_text_1,sample_text_2,sample_text_3]
no_docs=len(corp)
# print(no_docs)



# 对所有文档进行整数编码
# 所有唯一的单词都将由一个整数来表示。用的是keras的one-hot函数,请注意，指定的词汇大小足够大，以确保每个单词的整数编码都是唯一的。
# 注意一件重要的事情，即单词的整数编码在不同的文档中保持不变。比如：每一份文件中“butter”都用31表示。
vocab_size=50
encod_corp=[]
for i,doc in enumerate(corp):
    encod_corp.append(one_hot(doc,50))
    print("The encoding for document",i+1," is : ",one_hot(doc,50))
# print(encod_corp)





# 填充文档（使文档长度相同）
# Keras嵌入层要求所有单个文档的长度相同。因此，我们现在将用较短的文件填充0。因此，现在在keras嵌入层中，“输入长度”将等于最大长度或最大字数文档的长度（即字数）。
# 为了填充较短的文档，我使用了keras库的pad_sequences函数。
# length of maximum document. will be nedded whenever create embeddings for the words
maxlen=-1
for doc in corp:
    # 对句子进行分词
    tokens=nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen=len(tokens)
print("The maximum number of words in any document is : ",maxlen)

# now to create embeddings all of our docs need to be of same length. hence we can pad the docs with zeros.
pad_corp=pad_sequences(encod_corp,maxlen=maxlen,padding='post',value=0.0)
print("No of padded documents: ",len(pad_corp))
for i,doc in enumerate(pad_corp):
     print("The padded encoding for document",i+1," is : ",doc)






# 使用keras嵌入层实际创建嵌入
# 现在所有文档的长度都相同（填充后）。现在我们已经准备好创建和使用嵌入。
# 我会把这些词嵌入到8个维度的向量中。
# specifying the input shape
input=Input(shape=(no_docs,maxlen),dtype='float64')
'''
shape of input. 
each document has 12 element or words which is the value of our maxlen variable.

'''
word_input=Input(shape=(maxlen,),dtype='float64')
# creating the embedding
word_embedding=Embedding(input_dim=vocab_size,output_dim=8,input_length=maxlen)(word_input)
word_vec=Flatten()(word_embedding) # flatten
embed_model =Model([word_input],word_vec) # combining all into a Keras model







# 嵌入层参数---
# 'input_dim'=我们将选择的vocab大小。换言之，它是vocab中唯一单词的数目。
# 'output_dim'=希望嵌入的维度数。每一个词都将由这么多维度的向量表示。
# “input_length”=最大文档的长度。在我们的例子中，它存储在maxlen变量中。
embed_model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),loss='binary_crossentropy',metrics=['acc'])
# compiling the model. parameters can be tuned as always.
print(type(word_embedding))
print(word_embedding)
print(embed_model.summary()) # summary of the model



embeddings=embed_model.predict(pad_corp) # finally getting the embeddings.
print("Shape of embeddings : ",embeddings.shape)
print(embeddings)



embeddings=embeddings.reshape(-1,maxlen,8)
print("Shape of embeddings : ",embeddings.shape)
print(embeddings)



# The resulting shape is (3,12,8).
# 3---> no of documents
# 12---> each document is made of 12 words which was our maximum length of any document.
# & 8---> each word is 8 dimensional.



# 获取特定文档中特定单词的编码
for i,doc in enumerate(embeddings):
    for j,word in enumerate(doc):
        print("The encoding for ",j+1,"th word","in",i+1,"th document is : \n\n",word)





# 现在，这使得我们更容易看到我们有3个文档，每个文档由12个（maxlen）单词组成，每个单词映射到一个8维向量。

# 如何处理真正的文本
# 就像上面一样，我们现在可以使用任何其他文档。
# 每个句子都有一个单词列表，我们将使用'one_hot'函数进行整数编码，如下所示。
# 现在每个句子都会有不同的单词数。 所以我们需要用最大的单词将序列填充到句子中。
# 此时，我们已准备好将输入馈送到Keras嵌入层，如上所示。
# 'input_dim'=我们将选择的词汇大小
# 'output_dim'=我们希望嵌入的维度数
# 'input_length'=最大文档的长度