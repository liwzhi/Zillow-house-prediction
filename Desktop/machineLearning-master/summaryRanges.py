# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:46:30 2015

@author: weizhi
"""

class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if not nums:
            return []
        if len(nums)==1:
            return [str(nums[0])]
        start = 0
        end = len(nums)-1
        result = []
        while start<end:
            mid = start+1
            curr = nums[start]
            count = 0
            while mid<=end and nums[mid]-nums[start] ==1:
                start+=1
                mid +=1
                count +=1
            if count>0:
                string = str(curr) +'->' + str(nums[start])
            else:
                string = str(curr)
            print string
            result.append(string)
            start = mid
            if start == end:
                result.append(str(nums[end]))
            
        return result
            
                
                
        