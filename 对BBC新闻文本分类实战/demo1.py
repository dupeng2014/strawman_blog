# https://www.kaggle.com/yufengdev/bbc-text-categorization
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tensorflow import keras


data = pd.read_csv("./bbc-text.csv")
print(data.head())
print(data['category'].value_counts())


train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test


train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)


max_words = 1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words,
                                              char_level=False)


tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)


# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


#models_name = ["k-NN1", "k-NN2", "NB1", "LR-1", "LinearSVM",  "SGD", "DecisionTree", "RandomForest", "NeuralNet"]
models_name = ["Multi NB", "LR", "LinearSVM" ]

models = [ MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), LogisticRegression(), LinearSVC()]

for j in range(len(models)):
    models[j].fit(x_train, y_train)  # 利用训练数据对模型参数进行估计
    y_predict = models[j].predict(x_test)  # 对参数进行预测
    print(models_name[j], '：', accuracy_score(y_test, y_predict))

