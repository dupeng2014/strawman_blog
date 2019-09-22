
def getMid(input_list, left, right):
    mid = int((left+right)/2)
    if input_list[left] <= input_list[right]:
        if input_list[mid] < input_list[left]:
            return left
        elif input_list[mid] > input_list[right]:
            return right
        else:
            return mid
    else:
        if input_list[mid] < input_list[right]:
            return right
        elif input_list[mid] > input_list[left]:
            return left
        else:
            return mid



def division(input_list, left, right):
    '''
    函数说明:根据left和right进行一次扫描，重新找到基准数
    Parameters:
        input_list - 待排序列表
        left - 左指针位置
        right - 右指针位置
    Returns:
        left - 新的基准数位置
    '''


    base = getMid(input_list, left, right)
    input_list[base], input_list[right] = input_list[right], input_list[base]


    base = input_list[left]
    while left < right:
        while left < right and input_list[right] >= base:
            right -= 1
        input_list[left] = input_list[right]
        while left < right and input_list[left] <= base:
            left += 1
        input_list[right] = input_list[left]
    input_list[left] = base
    return left



def QuickSort(input_list, left, right):
    '''
    函数说明:快速排序（升序）
    Parameters:
        input_list - 待排序列表
    Returns:
        无
    '''
    if left < right:
        base_index = division(input_list, left, right)
        QuickSort(input_list, left, base_index - 1)
        QuickSort(input_list, base_index + 1, right)


if __name__ == '__main__':
    input_list = [6, 4, 8, 9, 2, 3, 1]
    print('排序前:', input_list)
    QuickSort(input_list, 0, len(input_list) - 1)
    print('排序后:', input_list)
