# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 00:16:38 2015

@author: weizhi
"""

class Solution(object):
    pre = 0
    flag = 0
    Dict = []
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        self.Dict.append(n)
        result = 0
        
        
        for item in str(n):
            result += int(item)**2
       # print result
        if result==1:
            return True
        #print self.Dict
        while(result!=1 and result not in self.Dict):
            self.isHappy(result)
            #print self.Dict

        return result==1
        
class Solution1(object):
    pre = 0
    flag = 1
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        while(n!=1 and self.pre!=n):
            self.pre = n
            n = self.transform(n)
          #  self.pre = n
        return n==1 or self.pre==n
    def transform(self,n):
        result = 0
        for item in str(n):
            result+=int(item)**2
        return result        
obj = Solution()
print obj.isHappy(19)