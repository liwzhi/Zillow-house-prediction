# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:06:59 2015

@author: weizhi
"""

class Solution:
    """
    @param nums: A list of integers
    @return: An integer denote the sum of maximum subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        maxValue = float('-inf')
        local = 0
        for item in nums:
            maxValue = max(local+item,maxValue)
            if local+item>=0:
                local = local + item
            else:
                local = 0
        return maxValue
obj = Solution()
print obj.maxSubArray([-2,2,-3,4,-1,2,1,-5,3])