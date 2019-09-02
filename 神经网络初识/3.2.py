# Mine
from numpy import exp, array, random, dot


class NeuralNetwork():

    def __init__(self):
        # 设置随机数种子，使每次运行生成的随机数相同
        # 便于调试
        random.seed(1)

        # 输入层三个神经元作为第一层
        # 第二层定义为5个神经元
        # 第三层定义为4个神经元
        layer2 = 5
        layer3 = 4

        # 随机初始化各层权重
        self.synaptic_weights1 = 2 * random.random((3, layer2)) - 1
        self.synaptic_weights2 = 2 * random.random((layer2, layer3)) - 1
        self.synaptic_weights3 = 2 * random.random((layer3, 1)) - 1

    # Sigmoid函数, 图像为S型曲线.
    # 我们把输入的加权和通过这个函数标准化在0和1之间。
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    # Sigmoid函数的导函数.
    # 即使Sigmoid函数的梯度
    # 它同样可以理解为当前的权重的可信度大小
    # 梯度决定了我们对调整权重的大小，并且指明了调整的方向
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # 我们通过不断的试验和试错的过程来训练神经网络
    # 每一次都对权重进行调整
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            # 正向传播过程，即神经网络“思考”的过程
            activation_values2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            activation_values3 = self.__sigmoid(dot(activation_values2, self.synaptic_weights2))
            output = self.__sigmoid(dot(activation_values3, self.synaptic_weights3))
            # 计算各层损失值
            delta4 = (training_set_outputs - output) * self.__sigmoid_derivative(output)
            delta3 = dot(self.synaptic_weights3, delta4.T) * (self.__sigmoid_derivative(activation_values3).T)
            delta2 = dot(self.synaptic_weights2, delta3) * (self.__sigmoid_derivative(activation_values2).T)

            # 计算需要调制的值
            adjustment3 = dot(activation_values3.T, delta4)
            adjustment2 = dot(activation_values2.T, delta3.T)
            adjustment1 = dot(training_set_inputs.T, delta2.T)

            # 调制权值
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3

    # 神经网络的“思考”过程
    def think(self, inputs):
        activation_values2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        activation_values3 = self.__sigmoid(dot(activation_values2, self.synaptic_weights2))
        output = self.__sigmoid(dot(activation_values3, self.synaptic_weights3))
        return output

if __name__ == "__main__":
    # 初始化
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("\nRandom starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("\nRandom starting synaptic weights (layer 3): ")
    print(neural_network.synaptic_weights3)

    # 训练集
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("\nNew synaptic weights (layer 1) after training: ")
    print(neural_network.synaptic_weights1)
    print("\nNew synaptic weights (layer 2) after training: ")
    print(neural_network.synaptic_weights2)
    print("\nNew synaptic weights (layer 3) after training: ")
    print(neural_network.synaptic_weights3)

    # 新样本测试
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0])))
