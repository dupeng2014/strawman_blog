def selectSort(input_list):
    '''
    选择排序
    :param input_list: 输入数组
    :return: 返回已排序的数组
    '''
    if len(input_list) == 0:
        return []
    sorted_list = input_list
    length = len(sorted_list)
    for i in range(length):
        min_index = i
        for j in range(i + 1, length):
            if sorted_list[min_index] > sorted_list[j]:
                min_index = j
        if min_index == i:
            continue
        temp = sorted_list[i]
        sorted_list[i] = sorted_list[min_index]
        sorted_list[min_index] = temp
    return sorted_list



if __name__ == '__main__':
    input_list = [6, 4, 8, 9, 2, 3, 1]
    print('排序前:', input_list)
    sorted_list = selectSort(input_list)
    print('排序后:', sorted_list)