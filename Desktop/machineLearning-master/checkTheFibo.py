# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:13:18 2015

@author: weizhi
"""

class Solution():
    def fibo(self,nums):
        dp = [1 for i in range(len(nums))]
        if not nums:
            return 
        for i in range(2,len(nums)):
            dp[i] = dp[i-1] + dp[i-2]
        print dp
        return dp==nums
obj = Solution()
print obj.fibo([1,1])
