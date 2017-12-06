# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:03:58 2015

@author: weizhi
"""

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices or len(prices)==1:
            return 0
        pre = 0
        curr = 1
        result = 0
        for index in range(1,len(prices)):
            curr = index
            pre = index-1
            if prices[curr]>prices[pre]:
                result +=prices[curr] - prices[pre]
        return result