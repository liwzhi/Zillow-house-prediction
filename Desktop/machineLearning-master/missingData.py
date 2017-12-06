# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:12:15 2015

@author: weizhi
"""
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 
        if min(nums)<0:
            return 
        
        sumN = sum(range(max(nums)+1))
        print sumN
        flag =0
        for item in nums:
            sumN = sumN - item
            if item ==0:
                flag = 1
                
        if sumN==0 and flag ==0:
            return 0
        elif sumN ==0:
            return max(nums)+ 1
        else:
            return sumN
obj = Solution()
result = obj.missingNumber([0,1,2])
print result

