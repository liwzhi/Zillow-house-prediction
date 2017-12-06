# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:39:45 2015

@author: weizhi
"""

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if not digits:
            return 
        result = 0
        for item in digits:
            result = result*10 + item
        return [int(item) for item in str(result+1)]
        
obj = Solution()
print obj.plusOne([1,0])
        