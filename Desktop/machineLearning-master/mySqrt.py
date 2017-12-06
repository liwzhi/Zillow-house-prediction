# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:54:28 2015

@author: weizhi
"""

class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==0:
            return 0
        left = 1
        right = x
      #  right = x/2 + 1
        while left<=right:
            mid = left + (right-left)/2
           # print mid
            if mid**2==x:
                return mid
            elif mid**2>x:
                right = mid-1
            else:
                left = mid+1
        return right
        
        
obj = Solution()
print obj.mySqrt(5)