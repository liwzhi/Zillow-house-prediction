# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:03:41 2015

@author: weizhi
"""

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return 0
        if len(nums)==1:
            return nums[0]
        if len(nums)<=2:
            return max(nums)
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = nums[1]
        for i in range(2,len(nums)):
            dp[i] = max(nums[i] + max(dp[:i-1]),dp[i-1])
        return dp[len(nums)-1]
obj = Solution()
print obj.rob([2,1,1,1])