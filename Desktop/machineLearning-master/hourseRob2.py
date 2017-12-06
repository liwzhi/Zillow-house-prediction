

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

        if len(nums)<=3:
            return max(nums)
        
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = nums[1]
        dp[2] = max(nums[1],nums[2])
        for i in range(3,len(nums)):
            dp[i] = max(dp[i-1],nums[i] + max(dp[1:i-1]))
        maxValue1 = dp[len(nums)-1]
        
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = nums[1]
        for i in range(2,len(nums)-1):
            dp[i] = max(dp[i-1],nums[i] + max(dp[:i-1]))
        maxValue2 = dp[len(nums)-2]
      #  print maxValue1
      #  print maxValue2
        return max(maxValue1,maxValue2)
obj = Solution()
print obj.rob([2,1,1,1])