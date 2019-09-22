def maxProfit(prices):
    '''
    只能交易一次
    :param prices: 它的第 i 个元素是一支给定股票第 i 天的价格
    :return: 最大收益
    '''

    '''
    动态规划：前i天的最大收益 = max{前i-1天的最大收益，第i天的价格 - 前i-1天中的最小价格}
    '''
    min_p, max_p = 999999, 0
    for i in range(len(prices)):
        min_p = min(min_p, prices[i])
        max_p = max(max_p, prices[i] - min_p)
    return max_p


if __name__ == '__main__':
    prices = [7, 1, 5, 3, 6, 4]
    print(maxProfit(prices))
