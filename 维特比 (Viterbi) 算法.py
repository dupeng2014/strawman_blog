# import numpy as np
# def viterbi(trainsition_probability,emission_probability,pi,obs_seq):
#     #转换为矩阵进行运算
#     trainsition_probability=np.array(trainsition_probability)
#     emission_probability=np.array(emission_probability)
#     pi=np.array(pi)
#     obs_seq = [0, 2, 3]
#     # 最后返回一个Row*Col的矩阵结果
#     Row = np.array(trainsition_probability).shape[0]
#     Col = len(obs_seq)
#     #定义要返回的矩阵
#     F=np.zeros((Row,Col))
#     #初始状态
#     F[:,0]=pi*np.transpose(emission_probability[:,obs_seq[0]])
#     for t in range(1,Col):
#         list_max=[]
#         for n in range(Row):
#             list_x=list(np.array(F[:,t-1])*np.transpose(trainsition_probability[:,n]))
#             #获取最大概率
#             list_p=[]
#             for i in list_x:
#                 list_p.append(i*10000)
#             list_max.append(max(list_p)/10000)
#         F[:,t]=np.array(list_max)*np.transpose(emission_probability[:,obs_seq[t]])
#     return F
#
# if __name__=='__main__':
#     #隐藏状态
#     invisible=['Sunny','Cloud','Rainy']
#     #初始状态
#     pi=[0.63,0.17,0.20]
#     #转移矩阵
#     trainsion_probility=[[0.5,0.375,0.125],[0.25,0.125,0.625],[0.25,0.375,0.375]]
#     #发射矩阵
#     emission_probility=[[0.6,0.2,0.15,0.05],[0.25,0.25,0.25,0.25],[0.05,0.10,0.35,0.5]]
#     #最后显示状态
#     obs_seq=[0,2,3]
#     #最后返回一个Row*Col的矩阵结果
#     F=viterbi(trainsion_probility,emission_probility,pi,obs_seq)
#     print(F)


import numpy as np
# -*- codeing:utf-8 -*-
__author__ = 'youfei'

#   隐状态
hidden_state = ['sunny', 'rainy']

#   观测序列
obsevition = ['walk', 'shop', 'clean']


#   根据观测序列、发射概率、状态转移矩阵、发射概率
#   返回最佳路径
def compute(obs, states, start_p, trans_p, emit_p):
    #   max_p（3*2）每一列存储第一列不同隐状态的最大概率
    max_p = np.zeros((len(obs), len(states)))

    #   path（2*3）每一行存储上max_p对应列的路径
    path = np.zeros((len(states), len(obs)))

    #   初始化
    for i in range(len(states)):
        max_p[0][i] = start_p[i] * emit_p[i][obs[0]]
        path[i][0] = i

    for t in range(1, len(obs)):
        newpath = np.zeros((len(states), len(obs)))
        for y in range(len(states)):
            prob = -1
            for y0 in range(len(states)):
                nprob = max_p[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]
                if nprob > prob:
                    prob = nprob
                    state = y0
                    #   记录路径
                    max_p[t][y] = prob
                    for m in range(t):
                        newpath[y][m] = path[state][m]
                    newpath[y][t] = y

        path = newpath

    max_prob = -1
    path_state = 0
    #   返回最大概率的路径
    for y in range(len(states)):
        if max_p[len(obs)-1][y] > max_prob:
            max_prob = max_p[len(obs)-1][y]
            path_state = y

    return path[path_state]


state_s = [0, 1]
obser = [0, 1, 2]

#   初始状态，测试集中，0.6概率观测序列以sunny开始
start_probability = [0.6, 0.4]

#   转移概率，0.7：sunny下一天sunny的概率
transititon_probability = np.array([[0.7, 0.3], [0.4, 0.6]])

#   发射概率，0.4：sunny在0.4概率下为shop
emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

result = compute(obser, state_s, start_probability, transititon_probability, emission_probability)

for k in range(len(result)):
    print(hidden_state[int(result[k])])
