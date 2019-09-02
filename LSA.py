from sklearn.decomposition import TruncatedSVD           # namely LSA/LSI(即潜在语义分析)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



# ♪ Until the Day ♪ by JJ Lin 林俊杰
docs = ["In the middle of the night",
        "When our hopes and fears collide",
        "In the midst of all goodbyes",
        "Where all human beings lie",
        "Against another lie"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names()
print(terms)

n_pick_topics = 3            # 设定主题数为3
lsa = TruncatedSVD(n_pick_topics)
X2 = lsa.fit_transform(X)
print(X2)
