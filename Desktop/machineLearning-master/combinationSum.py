# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:47:22 2015

@author: weizhi
"""

class Solution(object):
    def combination(self,candiates,target):
        res = []
        candiates.sort()
        self.combinationSum(candiates,target,0,[],res)
        return res
        
    
    def combinationSum(self,candiates,target,start,stk,res):
        if target ==0:
            res.append(stk)
        if target>0:
            for i in range(start,len(candiates)):
                self.combinationSum(candiates,target-candiates[i],i,stk+[candiates[i]],res)

Obj = Solution()
result = Obj.combination([2,3,6,7],9)
print result