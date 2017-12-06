# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:35:52 2015

@author: weizhi
"""

class Solution(object):
    def maxSubArray(self,nums):
        if len(nums) ==0:
            return 
        maxValue = float('-inf')
        localMax = 0
        for item in nums:
            localMax = max(0,localMax+item)
            maxValue = max(maxValue,localMax)
        return maxValue
obj = Solution()
print obj.maxSubArray([-1,2,3,-1,4,5,-1,-9])
