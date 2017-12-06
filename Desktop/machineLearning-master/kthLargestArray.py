# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:41:49 2015

@author: weizhi
"""

import heapq
class Solution:
    # @param {integer[]} nums
    # @param {integer} k
    # @return {integer}
    def findKthLargest(self, nums, k):
        if len(nums)==0:
            return 
        h = []
        for value in nums:
            heapq.heappush(h,value)
        print h
        return heapq.nlargest(k,h)[-1]
obj = Solution()
result = obj.findKthLargest([1,2,3,4,5,-1,-20,-30],4)
print result