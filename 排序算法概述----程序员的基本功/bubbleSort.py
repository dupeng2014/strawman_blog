def bubbleSort(input_list):
    '''
    冒泡排序
    :param input_list:list数组
    :return:list数组
    '''
    sort_list = input_list
    n = len(sort_list)

    # 遍历所有数组元素
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if sort_list[j] > sort_list[j + 1]:
                sort_list[j], sort_list[j + 1] = sort_list[j + 1], sort_list[j]

    return sort_list



if __name__ == '__main__':
    input_list = [1, 9, 2, 5, 4]
    sort_list = bubbleSort(input_list)
    print(sort_list)



