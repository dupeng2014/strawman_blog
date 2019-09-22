def LongestCommonSubstring(FirstString, SecondString):
    '''
    求最长子串解法2：
    建立一个以字符串1长度+1乘字符串2长度+1的矩阵
    矩阵中如果矩阵的行i对应的单词等于列j对应的单词，那么就在对应的m[i+1][j+1]位置等于m[i][j]+1
    再将m[i+1][j+1]与最大程度比较，从而找到最大值

    FirstString----字符串1
    SecondString---字符串2
    Longest--------最长公共子串的长度
    ComputeTimes--计算次数，用于对比计算量
    '''
    FirstStringLenght = len(FirstString)
    SecondStringLenght = len(SecondString)

    if (FirstStringLenght == 0) | (SecondStringLenght) == 0:
        return ''

    Longest = 0
    ComputeTimes = 0

    '''
    构建一个矩阵，行数为字符串1的长度+1，列数为字符串2的长度+1
    这里+1，是为了计算方便，如果不+1，我们需要单独对第一行，和第一列做一次循环，
    +1后，我们就可以舍去这段循环。
    主要是为了这个公式 m[i+1][j+1]=m[i][j]+1 服务
    '''

    m = [[0 for i in range(SecondStringLenght+1)] for j in range(FirstStringLenght+1)]


    for i in range(FirstStringLenght):
        for j in range(SecondStringLenght):
            if FirstString[i] == SecondString[j]:
                ComputeTimes += 1
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > Longest:
                    ComputeTimes += 1
                    Longest, SecondStringStartPoint = m[i + 1][j + 1], j + 1
    return Longest, SecondString[SecondStringStartPoint - Longest:SecondStringStartPoint], ComputeTimes




if __name__ =='__main__':
    print(LongestCommonSubstring('abcdefg', 'abcjfoiewajfiowejfdefg'))