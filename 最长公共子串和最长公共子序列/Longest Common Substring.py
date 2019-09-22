def find_longest_common_subsequence(x, y):

    maxLen = 0
    maxEnd = 0

    # 首先构造那个矩阵
    c = [[0 for i in range(len(y) + 1)] for j in range(len(x) + 1)]

    for i in range(len(c)):
        for j in range(len(c[1])):
            if i == 0 or j == 0:
                c[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = 0

            if c[i][j] > maxLen:
                maxLen = c[i][j]
                maxEnd = i #若记录i, 则最后获取LCS时是取str1的子串

    return c, maxLen, maxEnd



if __name__ == '__main__':
    x = ['A', 'B', 'C', 'B', 'D', 'A', 'B', 'Z', 'M', 'X', 'Y', 'H', 'A']
    y = ['B', 'D', 'C', 'A', 'Z', 'M', 'X', 'Y', 'H', 'B', 'C', 'B', 'A']

    c ,maxLen, maxEnd= find_longest_common_subsequence(x, y)
    for item in c:
        # 打印出我们所谓的那个矩阵
        print(item)

    print(x[maxEnd - maxLen : maxEnd])
