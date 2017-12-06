# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:41:31 2015

@author: weizhi
"""

class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if not candidates:
            return
        result = []
        self.helper(sorted(candidates),target,result,[],0)
        return result
    def helper(self,candidates,target,result,stk,start):
        if target==sum(stk[:]):
            result.append(stk[:])
            return result
        pre = float('-inf')
        for i in range(start,len(candidates)):
            item = candidates[i]
            if item<=target and sum(stk)<target and pre!=item:
                stk.append(item)
                self.helper(candidates,target,result,stk,i+1)
                stk.pop()
            pre = item
        