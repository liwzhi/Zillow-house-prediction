# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:54:17 2015

@author: weizhi
"""

class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if not nums:
            return 
        if len(nums)==1:
            if nums[0] == val:
                return 0
            else:
                return 1
        j = len(nums)-1
        index = 0
        count =0
        while(index<=j):
            item = nums[index]
            if item==val:
                count +=1
                while(j>=0 and nums[j]==val):
                    j-=1
                temp = nums[j]
                nums[j] = nums[index]
                nums[index] = temp
                j-=1
                index+=1
            else:
                index+=1
        print nums
        print count
        return len(nums[:len(nums)-count])
obj = Solution()
print obj.removeElement([3,3,5,5],5)