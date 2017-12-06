# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:05:59 2015

@author: weizhi
"""

class Solution(object):
    def grayCode(self,n):
        if n ==0:
            return 
        result = []
        stk = []
        result = self.helper(n,result,stk,0)
        return result
    def helper(self,n,result,stk,depth):
        if depth == n:
            result.append(stk)
            return result
      #  stk.append(0)
        self.helper(n,result,stk + [0],depth+1)
        self.helper(n,result,stk + [1],depth+1)
        #inter = [left,right]
        return result
obj = Solution()
result = obj.grayCode(4)
print result