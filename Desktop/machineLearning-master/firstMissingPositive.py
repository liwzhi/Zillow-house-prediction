# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:44:00 2015

@author: weizhi
"""

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 1
        list(set(nums))
        maxValue = max(nums)
        minValue = min(nums)
        if maxValue<0:
            return 1
        Sum = sum(range(maxValue+1))
       # print Sum
        for item in nums:
            if item >0:
                Sum -=item 
               # print item
        #print flag
        #print Sum
       # print Sum
        
        if Sum!=0:
            return Sum
        elif Sum==0:
            return maxValue +  1
        
obj = Solution()
result = obj.firstMissingPositive([1000,-1])
print result        
            