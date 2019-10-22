# 最简洁的一种方法
def comNum(nums1, nums2):
    '''
    两个list的公共list
    :param nums1:
    :param nums2:
    :return:公共list
    '''
    res = []
    for k in nums1:
        if k in nums2:
            res.append(k)
            # 因为题干要求了出现次数一致，所以如果list1某一项在list2里，那么list2要remove这一项，否则次数会不一致。
            nums2.remove(k)
    return res

# 进阶解法 ↑
# 使用双指针将两个列表中共同的元素抠下来，因为已经排序，所以遇到不同元素时数值小的那个列表的指针向前移动
def comNum2(nums1, nums2):
    '''
    两个list的公共list
    :param nums1:
    :param nums2:
    :return:公共list
    '''
    nums1.sort()
    nums2.sort()
    nums1_point = 0
    nums2_point = 0
    com = []
    while(nums1_point < len(nums1) and nums2_point < len(nums2)):
        if nums1[nums1_point] == nums2[nums2_point]:
            com.append(nums1[nums1_point])
            nums1_point += 1
            nums2_point += 1
        elif nums1[nums1_point] < nums2[nums2_point]:
            nums1_point += 1
        else:
            nums2_point += 1

    return com


if __name__ == '__main__':
    nums1 = [4,9,5,4,4]
    nums2 = [9,4,9,8,4]
    print(comNum(nums1, nums2))

    nums3 = [4,9,5,4,4]
    nums4 = [9,4,9,8,4]
    print(comNum2(nums3, nums4))


