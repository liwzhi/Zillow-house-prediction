# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:37:23 2015

@author: weizhi
"""

class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        len1 = len(pattern)
        len2 = len(str.split(' '))
        if len1!=len2:
            return False
        Dict1 = self.transform(pattern)
        Dict2 = self.transform(str.split(' '))
      #  print Dict1
       # print Dict2
        return sorted(Dict1.values())==sorted(Dict2.values())
        
    
    def transform(self,string):
        Dict = {}
        for index in range(len(string)):
            item = string[index]
            if item in Dict:
                Dict[item].append(index)
            else:
                Dict[item] = [index]
        return Dict
obj = Solution()
print obj.wordPattern('abba','dog cat cat fish')