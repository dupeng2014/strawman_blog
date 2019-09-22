def findByBinary(input_list, target):
    min = 0
    max = len(input_list)-1

    # 建立一个死循环，直到找到key
    while min <= max:
        # 得到中位数
        # 这里一定要加int，防止列表是偶数的时候出现浮点数据
        center = int((min + max) / 2)
        # key在数组左边
        if input_list[center] > target:
            max = center - 1  # key在数组右边
        elif input_list[center] < target:
            min = center + 1
        # key在数组中间
        elif input_list[center] == target:
            print(str(target) + "在数组里面的第" + str(center) + "个位置")
            # return input_list[center]
            return center
    return min

def insertSort(input_list):
    '''
    二分法插入排序
    :param input_list: input_list
    :return: sort_list
    '''
    sort_list = input_list
    n = len(sort_list)

    for i in range(1, n):
        index = findByBinary(input_list[0:i], input_list[i])
        j = i-1
        temp = sort_list[i]
        # 元素后移
        while j >= index:
            sort_list[j + 1] = sort_list[j]
            j -= 1
        sort_list[j + 1] = temp

    return sort_list

if __name__ == '__main__':
    input_list = [0, 1, 8, 9, 2, 3, 1, 100, 67]
    sort_list = insertSort(input_list)
    print(sort_list)