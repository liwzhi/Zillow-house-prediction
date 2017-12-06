# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:33:22 2015

@author: weizhi
"""

class Solution(object):
    def maxProduct(self,nums):
        if len(nums)==0:
            return 
        currMax = currMin = res = nums[0]
        for i in range(1,len(nums)):
            item = nums[i]
            temp = currMax
            currMax = max(max(currMax*item,currMin*item),item)
            currMin = min(min(temp*item,currMin*item),item)
            #print currMin
            res = max(res,currMax)
        return res
a = Solution()
print a.maxProduct([1,-2,9,3,-4])