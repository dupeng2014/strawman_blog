# https://github.com/ashwinmj/word-prediction/blob/master/MarkovModel.ipynb

# Preamble
import string
import numpy as np

# Path of the text file containing the training data
training_data_file = 'eminem_songs_lyrics.txt'

# 去除标点符号
def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','', string.punctuation))


# 单词字典
def add2dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)


# 将list中的数转换成概率（频率）----频数/总数
def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict

# 每个单词的初始概率，就是每个单词出现的概率（频率）
initial_word = {}
# 某一个单词后面，出现某个单词出现的概率（频率）
second_word = {}
# 某两个单词后面，出现某个单词的概率（频率）
transitions = {}


# Trains a Markov model based on the data in training_data_file
# 训练马尔可夫模型
def train_markov_model():
    for line in open(training_data_file):
        tokens = remove_punctuation(line.rstrip().lower()).split()
        tokens_length = len(tokens)

        for i in range(tokens_length):
            token = tokens[i]
            if i == 0:
                initial_word[token] = initial_word.get(token, 0) + 1
                # print(initial_word)
            else:
                prev_token = tokens[i - 1]
                if i == tokens_length - 1:
                    add2dict(transitions, (prev_token, token), 'END')
                if i == 1:
                    add2dict(second_word, prev_token, token)
                else:
                    prev_prev_token = tokens[i - 2]
                    add2dict(transitions, (prev_prev_token, prev_token), token)

    # Normalize the distributions
    initial_word_total = sum(initial_word.values())
    for key, value in initial_word.items():
        initial_word[key] = value / initial_word_total

    for prev_word, next_word_list in second_word.items():
        second_word[prev_word] = list2probabilitydict(next_word_list)

    for word_pair, next_word_list in transitions.items():
        transitions[word_pair] = list2probabilitydict(next_word_list)

    print('Training successful.')



train_markov_model()


# 随机抽取某个单词，测试时使用
def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for key, value in dictionary.items():
        cumulative += value
        if p0 < cumulative:
            return key
    assert(False)



number_of_sentences = 10

# Function to generate sample text
# 利用抽取的那个单词预测生成
def generate():
    for i in range(number_of_sentences):
        sentence = []
        # Initial word
        word0 = sample_word(initial_word)
        sentence.append(word0)
        # Second word
        word1 = sample_word(second_word[word0])
        sentence.append(word1)
        # Subsequent words untill END
        while True:
            word2 = sample_word(transitions[(word0, word1)])
            if word2 == 'END':
                break
            sentence.append(word2)
            word0 = word1
            word1 = word2
        print(' '.join(sentence))


generate()