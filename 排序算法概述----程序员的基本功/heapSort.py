def HeapAdjust(input_list, parent, length):
    '''
    堆调整，调整为最大堆
    Parameters:
        input_list - 待排序列表
        parent - 堆的父结点
        length - 数组长度
    Returns:
        无
    '''
    # temp保存当前父节点
    temp = input_list[parent]
    # 先获得左孩子
    child = 2 * parent + 1

    while child < length:
        # 如果有右孩子结点，并且右孩子结点的值大于左孩子结点，则选取右孩子结点
        if child + 1 < length and input_list[child] < input_list[child + 1]:
            child += 1
        if temp >= input_list[child]:
            break
        # 把孩子结点的值赋给父结点
        input_list[parent] = input_list[child]
        # 选取孩子结点的左孩子结点, 继续向下筛选
        parent = child
        child = 2 * parent + 1
    input_list[parent] = temp


def HeapSort(input_list):
    '''
    函数说明:堆排序（升序）
    Parameters:
        input_list - 待排序列表
    Returns:
        sorted_list - 升序排序好的列表
    '''

    if len(input_list) == 0:
        return []
    sorted_list = input_list
    length = len(sorted_list)

    # 创建初始堆
    # 若二叉树结点总数为n，则最后一个非叶子结点是第 ⌊n / 2⌋ 个。
    for i in range(0, length // 2)[::-1]:
        HeapAdjust(sorted_list, i, length)

    # [开始:结束:步进]
    for j in range(1, length)[::-1]:
        # 最后一个元素和第一元素进行交换
        temp = sorted_list[j]
        sorted_list[j] = sorted_list[0]
        sorted_list[0] = temp
        # 筛选R[0]结点，得到i - 1个结点的堆
        HeapAdjust(sorted_list, 0, j)
        print('第%d趟排序:' % (length - j), end='')
        print(sorted_list)

    return sorted_list



if __name__ == '__main__':
    input_list = [6, 4, 8, 9, 2, 3, 1]
    print('排序前:', input_list)
    sorted_list = HeapSort(input_list)
    print('排序后:', sorted_list)

