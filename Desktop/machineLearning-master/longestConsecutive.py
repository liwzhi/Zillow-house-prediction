# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:45:12 2015

@author: weizhi
"""

class Solution(object):
    def longestConsecutive(self, nums):
        numset, maxlen = set(nums), 0
        for n in nums:
            if n not in numset:
                continue
            currlen, tmp = 1, n+1
            while tmp in numset:
                currlen += 1
                numset.remove(tmp)
                tmp += 1
            tmp = n-1
            while tmp in numset:
                currlen += 1
                numset.remove(tmp)
                tmp -= 1
            maxlen = max(currlen, maxlen)
        return maxlen
        
class Solution1(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = sorted(set(nums))
        maxLen = 0
        if not nums:
            return maxLen
        pre = nums[0]
        currLen = 0
        for i in range(1,len(nums)):
            curr = nums[i]
            if curr-pre==1:
                currLen +=1
                pre = curr
            else:
                currLen = 0
                pre = curr
            maxLen = max(maxLen,currLen)
        return maxLen+1        
obj = Solution()
print obj.longestConsecutive([1,2,3,4,10,5,3,4,5,6,7,8])