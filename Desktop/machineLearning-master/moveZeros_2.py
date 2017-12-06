# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:45:27 2015

@author: weizhi
"""

class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums)==0:
            return 
        end = len(nums)-1
        start = 0
        while(start<end):
            curr = nums[start]
           # print curr
            endValue = nums[end]
            if curr ==0:
                while(endValue==0):
                    end -=1
                    endValue= nums[end]
                   # print endValue
             #   print curr
             #   temp = nums[start]
                nums[start] = nums[end]
                print endValue
                nums[end] = nums[start]
                start +=1
                end -=1
            else:
                start+=1
        return nums
obj = Solution()
print obj.moveZeroes([0,1,2,3,0,0,1])