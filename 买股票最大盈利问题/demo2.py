def maxProfit(prices):
    '''
    不限交易次数
    :param prices: 它的第 i 个元素是一支给定股票第 i 天的价格
    :return: 最大收益
    '''
    profit = 0
    for day in range(len(prices) - 1):
        differ = prices[day + 1] - prices[day]
        if differ > 0:
            profit += differ
    return profit


if __name__ == '__main__':
    prices = [7, 1, 5, 3, 6, 4]
    print(maxProfit(prices))