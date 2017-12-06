# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:42:32 2015

@author: weizhi
"""

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if len(citations)==0:
            return 0

        result = 0
        citations.sort()
        
        for index in range(len(citations)):
            result = max(result,min(citations[index],len(citations)-index))
            
        return result 
Obj = Solution()
print Obj.hIndex([5,3,6,1,2])