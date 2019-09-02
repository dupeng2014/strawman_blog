


#
# # 构建模型
# from keras.models import Sequential
# from keras.layers import Dense
#
# # 构建模型
# model = Sequential()
# model.add(Dense(2,input_shape=(1,)))
# model.add(Dense(1))



# # 多层感知器MLP模型
#
# from keras.models import Model
# from keras.layers import Input,Dense
# from keras.utils import plot_model
#
#
# import matplotlib.pyplot as plt
# from IPython.display import Image
#
# mnist_input = Input(shape=(784,),name='input')
# hidden1 = Dense(512,activation='relu',name='hidden1')(mnist_input)
# hidden2 = Dense(216,activation='relu',name='hidden2')(hidden1)
# hidden3 = Dense(128,activation='relu',name='hidden3')(hidden2)
# output = Dense(10,activation='softmax',name='output')(hidden3)
#
# model = Model(inputs=mnist_input,outputs=output)
#
# # 打印网络结构
# model.summary()
#
# # 产生网络拓补图
# # plot_model(model, to_file='multilayer_perceptron_graph.png')




# # 卷积神经网络(CNN)
# from keras.models import Model
# from keras.layers import Input,Dense
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPool2D
# from keras.utils import plot_model
# from IPython.display import Image
#
# mnist_input = Input(shape=(28,28,1), name='input')
#
# conv1 = Conv2D(128,kernel_size=4,activation='relu',name='conv1')(mnist_input)
# pool1 = MaxPool2D(pool_size=(2,2),name='pool1')(conv1)
#
# conv2 = Conv2D(64,kernel_size=4,activation='relu',name='conv2')(pool1)
# pool2 = MaxPool2D(pool_size=(2,2),name='pool2')(conv2)
#
# hidden1 = Dense(64,activation='relu',name='hidden1')(pool2)
# output = Dense(10,activation='softmax',name='output')(hidden1)
# model = Model(inputs=mnist_input,outputs=output)
#
# # 打印网络结构
# model.summary()




# # 递归神经网络(RNN)
# from keras.models import Model
# from keras.layers import Input,Dense
# from keras.layers.recurrent import LSTM
#
#
# mnist_input = Input(shape=(784,1),name='input') # 把每一个像素想成是一序列有前后关系的time_steps
# lstm1 = LSTM(128,name='lstm')(mnist_input)
# hidden1 = Dense(128,activation='relu',name='hidden1')(lstm1)
# output = Dense(10,activation='softmax',name='output')(hidden1)
# model = Model(inputs=mnist_input,outputs=output)
#
# # 打印网络结构
# model.summary()







# # 共享输入层
# from keras.models import Model
# from keras.layers import Input,Dense,Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPool2D
# from keras.layers.merge import concatenate
# from keras.utils import plot_model
#
# # 输入层
# mnist_input = Input(shape=(28,28,1),name='input')
#
# # 第一个特征提取层
# conv1 = Conv2D(32,kernel_size=4,activation='relu',name='conv1')(mnist_input) # <- 看这里
# pool1 = MaxPool2D(pool_size=(2,2),name='pool1')(conv1)
# flat1 = Flatten()(pool1)
#
# # 第二个特征提取层
# conv2 = Conv2D(16,kernel_size=8,activation='relu',name='conv2')(mnist_input) # <- 看这里
# pool2 = MaxPool2D(pool_size=(2,2),name='pool2')(conv2)
# flat2 = Flatten()(pool2)
#
# # 把这两个特征提取层的结果拼接起来
# merge = concatenate([flat1,flat2])
#
# # 进行全连接层
# hidden1 = Dense(64,activation='relu',name='hidden1')(merge)
#
# # 输出层
# output = Dense(10,activation='softmax',name='output')(hidden1)
#
# # 以model来组合整个网络
# model = Model(inputs=mnist_input,outputs=output)
#
# # 打印网络结构
# model.summary()





#
# from keras.models import Model
# from keras.layers import Input,Dense
# from keras.layers.recurrent import LSTM
# from keras.layers.merge import concatenate
# from keras.utils import plot_model
#
# # 输入层
# mnist_input = Input(shape=(784,1),name='input') # 把每一个像素想成是一序列有前后关系的time_steps
#
# # 特征提取层
# extract1 = LSTM(128,name='lstm1')(mnist_input)
#
# # 第一个解释层(浅层单连通层)
# interp1 = Dense(10,activation='relu',name='interp1')(extract1) # <- 看这里
#
# # 第二个解释层(深层3层模型)
# interp21 = Dense(64,activation='relu',name='interp21')(extract1) # <- 看这里
# interp22 = Dense(32,activation='relu',name='interp22')(interp21)
# interp23 = Dense(10,activation='relu',name='interp23')(interp22)
#
# # 把两个特征提取层的结果拼起来
# merge = concatenate([interp1,interp23],name='merge')
#
# # 输出层
# output = Dense(10,activation='softmax',name='output')(merge)
#
# # 以Ｍodel来组合整个网络
# model = Model(inputs=mnist_input,outputs=output)
#
# # 打印网络结构
# model.summary()



#
# # 多输入模型
# from keras.models import Model
# from keras.layers import Input,Dense,Flatten
# from keras.layers import Conv2D
# from keras.layers import MaxPool2D
# from keras.layers.merge import concatenate
# from keras.utils import plot_model
# from IPython.display import Image
#
# import os
# os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'  # 安装graphviz的路径
#
# # 第一个输入层
# img_gray_bigsize = Input(shape=(64,64,1),name='img_gray_bigsize')
# conv11 = Conv2D(32,kernel_size=4,activation='relu',name='conv11')(img_gray_bigsize)
# pool11 = MaxPool2D(pool_size=(2,2),name='pool11')(conv11)
# conv12 = Conv2D(16,kernel_size=4,activation='relu',name='conv12')(pool11)
# pool12 = MaxPool2D(pool_size=(2,2),name='pool12')(conv12)
# flat1 = Flatten()(pool12)
#
# # 第二个输入层
# img_rgb_smallsize = Input(shape=(32,32,3),name='img_rgb_bigsize')
# conv21 = Conv2D(32,kernel_size=4,activation='relu',name='conv21')(img_rgb_smallsize)
# pool21 = MaxPool2D(pool_size=(2,2),name='pool21')(conv21)
# conv22 = Conv2D(16,kernel_size=4,activation='relu',name='conv22')(pool21)
# pool22 = MaxPool2D(pool_size=(2,2),name='pool22')(conv22)
# flat2 = Flatten()(pool22)
#
# # 把两个特征提取层的结果拼起来
# merge = concatenate([flat1,flat2])
#
# # 用隐藏的全连接层来解释特征
# hidden1 = Dense(128,activation='relu',name='hidden1')(merge)
# hidden2 = Dense(64,activation='relu',name='hidden2')(hidden1)
#
# # 输出层
# output = Dense(10,activation='softmax',name='output')(hidden2)
# # 以Model来组合整个网络
# model = Model(inputs=[img_gray_bigsize,img_rgb_smallsize],outputs=output)
#
# # 打印网络结构
# model.summary()




# 多输出模型
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model

# 输入层
mnist_input = Input(shape=(784,1),name='input') # 吧每一个像素想成是一序列有前后关系的time_steps

# 特征提取层
extract = LSTM(64,return_sequences=True,name='extract')(mnist_input)

# 分类输出
class11 = LSTM(32,name='class11')(extract)
class12 = Dense(32,activation='relu',name='class12')(class11)
output1 = Dense(10,activation='softmax',name='output1')(class12)

# 序列输出
output2 = TimeDistributed(Dense(10,activation='softmax'),name='output2')(extract)

# 以Model来组合整个网络
model = Model(inputs=mnist_input,outputs=[output1,output2])

# 打印网络结构
model.summary()
