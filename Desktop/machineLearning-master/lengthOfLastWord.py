# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:53:19 2015

@author: weizhi
"""

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        return len(s.strip().split(' ')[-1])
        
obj = Solution()
print obj.lengthOfLastWord('a ')