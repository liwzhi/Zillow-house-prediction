# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:01:09 2015

@author: weizhi
"""

class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if a ==None or b==None:
            return 
        return bin(self.transform(a)+self.transform(b))[2:]
       # return bin(int(a)+int(b))[:2]
    def transform(self,s):
        result = 0
        count = 0
        for item in reversed(s):
            result = result + int(item)*2**(count)
            count +=1
        return result
obj = Solution()
print obj.addBinary('0','0')