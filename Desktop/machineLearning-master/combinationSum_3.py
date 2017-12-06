# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:01:02 2015

@author: weizhi
"""

class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        if n==0:
            return[]
        result = []
        candidates = range(1,10)
        result = self.helper(candidates,result,[],k,n,0)
        return result
    def helper(self,candidates,result,stk,k,n,start):
        if len(stk)==k and sum(stk[:])==n:
            result.append(stk[:])
            return result
        for i in range(start,len(candidates)):
            item = candidates[i]
            stk.append(item)
            if len(stk)<=k and sum(stk[:])<=n:
                self.helper(candidates,result,stk,k,n,i+1)
            stk.pop()
        return result
