from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout
from keras.models import Model
# from keras_contrib.layers import CRF


def create_model(maxlen, chars, word_size, infer=False):
    """

    :param infer:
    :param maxlen:
    :param chars:
    :param word_size:
    :return:
    """
    # print(chars)
    # print(len(chars))
    # exit()
    sequence = Input(shape=(maxlen,), dtype='int32')
    # 输入层的维度只能比总的字数大，大或等于0的整数，字典长度，即输入数据最大下标+1
    # 比如单词表大小为1000，词向量的维度为300，所以Embedding的参数 input_dim=10000，output_dim=300
    embedded = Embedding(input_dim=len(chars) + 1, output_dim=word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    model = Model(input=sequence, output=output)
    model.summary()
    if not infer:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

