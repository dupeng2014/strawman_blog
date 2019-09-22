def insertSort(input_list):
    '''
    直接插入排序
    :param input_list: input_list
    :return: sort_list
    '''
    sort_list = input_list
    n = len(sort_list)

    for i in range(1, n):
        temp = sort_list[i]
        j = i - 1
        while j >=0 and temp < sort_list[j]:
            sort_list[j + 1] = sort_list[j]
            j -= 1
        sort_list[j + 1] = temp
    return sort_list

if __name__ == '__main__':
    input_list = [6, 4, 8, 9, 2, 3, 1]
    sort_list = insertSort(input_list)
    print(sort_list)