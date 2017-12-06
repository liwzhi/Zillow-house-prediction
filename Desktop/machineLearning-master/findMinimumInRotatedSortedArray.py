# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:24:55 2015

@author: weizhi
"""

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        while left<right:
            mid = left + (right-left)/2
            if nums[left]<nums[mid]<nums[right]:
                return nums[left]
            elif nums[right]<nums[left]<nums[mid]:
                left = mid
            elif nums[mid]<nums[right]<nums[left]:
                right = mid
            else:
                break 
        return min(nums[left],nums[right])