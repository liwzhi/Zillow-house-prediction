# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:58:09 2015

@author: weizhi
"""

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        numsSet = set(nums)
        if not nums:
            return 1
        leftResult = float('-inf')
        rightResult = float('inf')
        for item in nums:
            temp = item+1
            resultRight = temp
            while temp in numsSet:
                numsSet.remove(temp)
                temp+=1
                resultRight = temp +1
                print resultRight

            temp = item -1
            resultLeft = max(1,temp)
            while temp in numsSet and temp>0:
                resultLeft = temp
                numsSet.remove(temp)
                temp-=1

            resultLeft = max(1,temp)
            left = max(leftResult,resultLeft)
            right = min(rightResult,resultRight)
        return left
obj = Solution()
print obj.firstMissingPositive([1,2,3,4,-1])