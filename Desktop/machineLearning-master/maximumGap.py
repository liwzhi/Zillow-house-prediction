# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:07:03 2015

@author: weizhi
"""

class Solution(object):
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:
            return 0
        nums = sorted(nums)
        curr = nums[1]-nums[0]
        for i in range(2,len(nums)):
            diff = nums[i]-nums[i-1]
            curr = max(curr, diff)
        return curr
            
            
Obj = Solution()
print Obj.maximumGap([100,1,2,3,4,5])