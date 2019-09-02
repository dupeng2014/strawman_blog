


# encoding=utf-8
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("精确模式/Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("全模式: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut_for_search("我来到北京清华大学")  # 搜索引擎模式
print("搜索引擎模式: " + "/ ".join(seg_list))  # 全模式


seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))




seg_list = jieba.cut("李小福是创新办主任也是云计算方面的专家")
print("之前: " + "/ ".join(seg_list))  # 全模式
jieba.load_userdict("user_dict.txt")
seg_list = jieba.cut("李小福是创新办主任也是云计算方面的专家")
print("之后: " + "/ ".join(seg_list))  # 全模式





print('==========关键词提取==========')

from jieba import analyse
# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags

# 原始文本
text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

# 基于TF-IDF算法进行关键词抽取
keywords = tfidf(text)
print("keywords by tf-idf:")
# 输出抽取出的关键词
print(keywords)




from jieba import analyse
# 引入TextRank关键词抽取接口
textrank = analyse.textrank

# 原始文本
text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

print("\nkeywords by textrank:")
# 基于TextRank算法进行关键词抽取
keywords = textrank(text)
# 输出抽取出的关键词
print(keywords)





print('\n')
print('==============词性标注================')

import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))






print('-----------')

result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))





print('\n\n\n')
print('==============nltk===================')

import nltk
text = 'PathonTip.com is a very good website. We can learn a lot from it.'
#将文本拆分成句子列表
sens = nltk.sent_tokenize(text)
print(sens)
#将句子进行分词,nltk的分词是句子级的,因此要先分句,再逐句分词,否则效果会很差.
words = []
for sent in sens:
    words.append(nltk.word_tokenize(sent))
print(words)




# 词性标注
text = 'PathonTip.com is a very good website'
#词性标注要利用上一步分词的结果
print(nltk.pos_tag(nltk.word_tokenize(text)))





# 命名实体识别
print('==命名实体识别===')
import nltk
text = 'Xi is the chairman of China in the year 2013.'
#分词
tokens = nltk.word_tokenize(text)
#词性标注
tags = nltk.pos_tag(tokens)
print(tags)
#NER需要利用词性标注的结果
ners = nltk.ne_chunk(tags)
print(ners)





print('======句子结构树=======')
# tree1 = nltk.Tree('NP',['Alick'])
# print(tree1)
# tree1.draw()
# tree2 = nltk.Tree('N',['Alick','Rabbit'])
# print(tree2)
# tree2.draw()
# tree3 = nltk.Tree('S',[tree1,tree2])
# print(tree3.label()) #查看树的结点
# tree3.draw()



words = ['table', 'probably', 'wolves', 'playing',
         'is', 'dog', 'the', 'beaches', 'grounded',
         'dreamt', 'envision']
print(words)

print("----------词干提取-------------")
# 在名词和动词中，除了与数和时态有关的成分以外的核心成分。
# 词干并不一定是合法的单词

pt_stemmer = nltk.stem.porter.PorterStemmer()  # 波特词干提取器
lc_stemmer = nltk.stem.lancaster.LancasterStemmer()   # 兰卡斯词干提取器
sb_stemmer = nltk.stem.snowball.SnowballStemmer("english")# 思诺博词干提取器

for word in words:
    pt_stem = pt_stemmer.stem(word)
    lc_stem = lc_stemmer.stem(word)
    sb_stem = sb_stemmer.stem(word)
    print("%8s %8s %8s %8s" % (word,pt_stem,lc_stem,sb_stem))







print('=============spacy==================')
import spacy
text = "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline."
nlp = spacy.load('en')
doc = nlp(text)


# 分句
for token in doc.sents:
    print(token)

# 词性标注
tags = [(token.text, token.tag_) for token in doc]
print(tags)


# 命名实体识别
for ent in doc.ents:
    print(ent.text, ent.label_)



# 依存关系分析、
nlp = spacy.load('en')
doc = nlp( "spaCy uses the terms head and child to describe the words" )
for token in doc:
    print('{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))


# 生成名词短语
doc = nlp('spaCy uses the terms head and child to describe the words')
for np in doc.noun_chunks:
    print(np)
    print(np.text, np.root.dep_, np.root.head.text)




print('\n\n\n\n')
print('============gensim==================')
import gensim
from gensim import corpora
from pprint import pprint
# How to create a dictionary from a list of sentences?
documents = [" Zoe Telford  played the police officer girlfriend of Simon, Maggie.",
             "Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again.",
             "Phoebe Thomas played Cheryl Cassidy, Paulines friend and also a year 11 pupil in Simons class.",
             "Dumped her boyfriend following Simons advice after he wouldnt ",
             "have sex with her but later realised this was due to him catching crabs off her friend Pauline."]




# 把句子分词
texts = [[text for text in doc.split()] for doc in documents]
print(texts)
# Create dictionary
dictionary = corpora.Dictionary(texts)
# Get information about the dictionary
print(dictionary)
# Show the word to id map
print(dictionary.token2id)
# 将每个句子中的单词排序，然后按(x, y)的形式输出，其中，x表示单词在词典中的序号 ，y表示出现的次数
corpus = [dictionary.doc2bow(text) for text in texts]
for cor in corpus:
    print(cor)


tfidf_model = gensim.models.TfidfModel(corpus)
print(tfidf_model)



# texts = [['human', 'interface', 'computer'],
# ['survey', 'user', 'computer', 'system', 'response', 'time'],
# ['eps', 'user', 'interface', 'system'],
# ['system', 'human', 'system', 'eps'],
# ['user', 'response', 'time'],
# ['trees'],
# ['graph', 'trees'],
# ['graph', 'minors', 'trees'],
# ['graph', 'minors', 'survey']]
#
# from gensim import corpora
# dictionary = corpora.Dictionary(texts)
# print(dictionary)
# print(dictionary.token2id)
# corpus = [dictionary.doc2bow(text) for text in texts]
# for cor in corpus:
#     print(cor)

