def bubbleSort(input_list):
    '''
    冒泡排序的一种优化
    :param input_list: input_list
    :return: sort_list
    '''
    sort_list = input_list
    n = len(sort_list)
    for i in range(n):
        exchanged = False
        for j in range(n-i-1):
            if sort_list[j] > sort_list[j+1]:
                sort_list[j], sort_list[j+1] = sort_list[j+1], sort_list[j]
                exchanged = True
        if not exchanged:
            break
    return sort_list





if __name__ == '__main__':
    input_list = [5, 1, 2, 3, 4]
    sort_list = bubbleSort(input_list)
    print(sort_list)