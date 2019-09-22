def find_longest_common_subsequence(x, y):
    # 首先构造那个矩阵
    c = [[0 for i in range(len(y) + 1)] for j in range(len(x) + 1)]

    # 这里还得构造个矩阵，为了输出公共子序列
    singal = [[' ' for i in range(len(y))] for j in range(len(x))]

    for i in range(len(c)):
        for j in range(len(c[1])):
            if i == 0 or j == 0:
                c[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
                singal[i - 1][j - 1] = '左上'
            else:
                c[i][j] = max(c[i - 1][j], c[i][j - 1])
                if c[i][j] == c[i - 1][j]:
                    singal[i - 1][j - 1] = '向上'
                else:
                    singal[i - 1][j - 1] = '向左'

    return c, singal


def show(singal, x, y):
    longest_common_subsequence = []
    i = len(x) - 1
    j = len(y) - 1
    while i > 0 or j > 0:
        if singal[i][j] == "向上":
            i -= 1
        elif singal[i][j] == '向左':
            j -= 1
        else:
            longest_common_subsequence.append(x[i])
            # 这里添加为y[j] 也可以
            i -= 1
            j -= 1
    longest_common_subsequence = list(reversed(longest_common_subsequence))  # 将列表进行逆转

    return longest_common_subsequence


if __name__ == '__main__':
    x = ['A', 'B', 'C', 'B', 'D', 'A', 'B']
    y = ['B', 'D', 'C', 'A', 'B', 'A']

    c, singal = find_longest_common_subsequence(x, y)
    for item in c:
        # 打印出我们所谓的那个矩阵
        print(item)
    for s in singal:
        print(s)

    # 找公共子序列
    longest_common_subsequence = show(singal, x, y)

    print("最大公共子序列:\n", longest_common_subsequence)

