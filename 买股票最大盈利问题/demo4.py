def greedy(prices):
    '''
    处理k过大导致超时的问题，用贪心解决
    当k大于数组长度的一半时，等同于不限次数交易
    :param prices:
    :return:
    '''
    res = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            res += prices[i] - prices[i - 1]
    return res

def maxProfit(k, prices):
    '''
    交易次数最多不能超过k次
    :param prices: 它的第 i 个元素是一支给定股票第 i 天的价格
    :return: 最大收益
    '''
    if not prices or not k:
        return 0
    n = len(prices)

    # 当k大于数组长度的一半时，等同于不限次数交易，用贪心算法解决，否则LeetCode会超时，也可以直接把超大的k替换为数组的一半，就不用写额外的贪心算法函数
    if k > n // 2:
        return greedy(prices)

    dp, res = [[[0] * 2 for _ in range(k + 1)] for _ in range(n)], []
    # dp[i][k][0]表示第i天已交易k次时不持有股票 dp[i][k][1]表示第i天已交易k次时持有股票
    # 设定在卖出时加1次交易次数
    for i in range(k + 1):
        dp[0][i][0], dp[0][i][1] = 0, - prices[0]
    for i in range(1, n):
        for j in range(k + 1):
            if not j:
                dp[i][j][0] = dp[i - 1][j][0]
            else:
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j - 1][1] + prices[i])
            dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i])
    # 「所有交易次数最后一天不持有股票」的集合的最大值即为问题的解
    for m in range(k + 1):
        res.append(dp[n - 1][m][0])
    return max(res)


if __name__ == '__main__':
    prices = [7, 1, 5, 3, 6, 4]
    print(maxProfit(1, prices))