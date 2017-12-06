# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:25:14 2015

@author: weizhi
"""

# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = n-1
        l = 0
        while(l<=r):
            mid = l + (r-l)/2
            if isBadVersion(mid)==False:
                l = mid+1
            else:
                r = mid-1
        return l
        