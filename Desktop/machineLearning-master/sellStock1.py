class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        minValue = prices[0]
        maxValue = prices[0]
        diffLocal = 0
        diff = 0
        for i in range(1,len(prices)):
            item = prices[i]
            minValue = min(minValue,item)
            #maxValue = max（maxValue,item)
            localDiff =item -minValue
            diff = max(localDiff,diff)
        return diff
        