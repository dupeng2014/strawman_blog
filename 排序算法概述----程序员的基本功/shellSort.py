def shellSort(input_list):
    length = len(input_list)
    if length <= 1:
        return input_list
    sorted_list = input_list
    # 划分组的个数
    gap = length // 2
    while gap > 0:
        # 为什么是 gap到length，因为第一个数不用插入，从第二个数才开始插入到第一个数
        for i in range(gap, length):
            # 划分后的每个数组开头index
            j = i - gap
            temp = sorted_list[i]
            while j >= 0 and temp < sorted_list[j]:
                sorted_list[j+gap] = sorted_list[j]
                j -= gap
            sorted_list[j+gap] = temp
        gap //= 2
    return sorted_list




if __name__ == '__main__':
    input_list = [13, 14, 94, 33, 82, 25, 59, 94, 65, 23, 45, 27, 73, 25, 39, 10]
    sort_list = shellSort(input_list)
    print(sort_list)