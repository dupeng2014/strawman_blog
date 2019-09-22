def maxProfit(prices):
    '''
    最多交易2次
    :param prices: 它的第 i 个元素是一支给定股票第 i 天的价格
    :return: 最大收益
    '''
    n = len(prices)
    if n < 2:
        return 0
    dp1 = [0 for _ in range(n)]
    dp2 = [0 for _ in range(n)]
    minval = prices[0]
    maxval = prices[-1]
    # 前向
    for i in range(1, n):
        dp1[i] = max(dp1[i - 1], prices[i] - minval)
        minval = min(minval, prices[i])
    # 后向
    for i in range(n - 2, -1, -1):
        dp2[i] = max(dp2[i + 1], maxval - prices[i])
        maxval = max(maxval, prices[i])

    dp = [dp1[i] + dp2[i] for i in range(n)]
    return max(dp)



def maxProfit_2(prices):
    '''
    标准的三维DP动态规划，三个维度，第一维表示天，第二维表示交易了几次，第三维表示是否持有股票。
    :param prices:
    :return:
    '''

    if not prices:
        return 0
    n = len(prices)
    dp = [[[0] * 2 for _ in range(3)] for _ in range(n)]
    # dp[i][j][0]表示第i天交易了j次时不持有股票, dp[i][j][1]表示第i天交易了j次时持有股票
    # 定义卖出股票时交易次数加1
    for i in range(3):
        dp[0][i][0], dp[0][i][1] = 0, -prices[0]

    for i in range(1, n):
        for j in range(3):
            if not j:
                dp[i][j][0] = dp[i - 1][j][0]
            else:
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j - 1][1] + prices[i])
            dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i])

    return max(dp[n - 1][0][0], dp[n - 1][1][0], dp[n - 1][2][0])



if __name__ == '__main__':
    prices = [7, 1, 5, 3, 6, 4]
    print(maxProfit(prices))
    print(maxProfit_2(prices))