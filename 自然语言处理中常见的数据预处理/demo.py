import re
temp = "想做/ 兼_职/学生_/ 的 、加,我Q：  1 5.  8 0. ！！？？  8 6 。0.  2。 3     有,惊,喜,哦"
string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", temp)
print(string)


pre_str = 'I Love My Family'
after_str = pre_str.lower()
print(after_str)



text = 'how are you, fine!'
words = text.split()
print(words)


from nltk.tokenize import word_tokenize
words = word_tokenize(text)
print(words)



from nltk.corpus import stopwords
print(stopwords.words('english'))
# 这里，NLTK 是基于特定的文本语料库或文本集。不同的语料库可能有不同的停止词。在一个应用中， 某个词可能是停止词。而在另一个应用这个词就是有意义的词。要从文本中清除停止词，可以使用带过滤条件的 Python 列表理解。
words = [w for w in words if w not in stopwords.words('english')]
print(words)


from nltk import pos_tag
sentence = word_tokenize('I always lie down to tell a lie.')
print(pos_tag(sentence))




from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
print(ne_chunk(pos_tag(word_tokenize('Antonio joined Udacity Inc. in California.'))))


words = ['branching', 'branched', 'branches']
from nltk.stem.porter import PorterStemmer
stemmed = [PorterStemmer().stem(w) for w in words]
print(stemmed)
















