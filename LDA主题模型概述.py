# 准备文档集合
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# 整合文档数据
doc_complete = [doc1, doc2, doc3, doc4, doc5]




# 数据清洗和预处理
# 数据清洗对于任何文本挖掘任务来说都非常重要，在这个任务中，移除标点符号，停用词和标准化语料库（Lemmatizer，对于英文，将词归元）。
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]



# 准备 Document - Term 矩阵
# 语料是由所有的文档组成的，要运行数学模型，将语料转化为矩阵来表达是比较好的方式。
# LDA 模型在整个 DT 矩阵中寻找重复的词语模式。
# Python 提供了许多很好的库来进行文本挖掘任务，“genism” 是处理文本数据比较好的库。下面的代码掩饰如何转换语料为 Document - Term 矩阵：
import gensim
from gensim import corpora
# 创建语料的词语词典，每个单独的词语都会被赋予一个索引
dictionary = corpora.Dictionary(doc_clean)
# 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# 构建 LDA 模型
# 创建一个 LDA 对象，使用 DT 矩阵进行训练。
# 训练需要上面的一些超参数，gensim 模块允许 LDA 模型从训练语料中进行估计，并且从新的文档中获得对主题分布的推断。
# 使用 gensim 来创建 LDA 模型对象
Lda = gensim.models.ldamodel.LdaModel
# 在 DT 矩阵上运行和训练 LDA 模型
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# 打印主题 每个主题含4个单词
print(ldamodel.print_topics(num_topics=3, num_words=3))

# [(0, '0.076*"sugar" + 0.076*"father" + 0.076*"sister"'), (1, '0.050*"spends" + 0.050*"time" + 0.050*"practice"'), (2, '0.065*"driving" + 0.065*"pressure" + 0.064*"stress"')]
# 每一行包含了主题词和主题词的权重，主题1 可以看作为“不良健康习惯”，
# 主题2 可以看作"工作效率"
# Topic3 可以看作 “家庭”。