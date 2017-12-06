# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:03:58 2015

@author: weizhi
"""

class Solution(object):
    def canJump(self,nums):
        if not nums:
            return False
        pre = nums[0]
        
     #   result = [0]*len(nums)
      #  result[0] = nums[0]
        for index in range(1,len(nums)):
            if pre>0:
                pass
            else:
                return False
            curr = nums[index] 
            if pre-1>=curr:
                pre = pre-1
            else:
                pre = curr

        return True