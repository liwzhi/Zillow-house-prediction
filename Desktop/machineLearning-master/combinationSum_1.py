# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:25:23 2015

@author: weizhi
"""


class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum(self, candidates, target):
        result = []
        self.combinationSumRecu(sorted(candidates), result, 0, [], target)
        return result
    
    def combinationSumRecu(self, candidates, result, start, intermediate, target):
        if target<0:
            return
        if target == 0:
            result.append(intermediate[:])
            return 
        for i in range(start,len(candidates)):
            if target<candidates[i]:
                return 
            print i
           # intermediate.append(candidates[i])
            self.combinationSumRecu(candidates,result,i,intermediate + [candidates[i]], target-candidates[i])
           # intermediate.pop()
#        while start < len(candidates) and candidates[start] <= target:
#            intermediate.append(candidates[start])
#            print start
#            self.combinationSumRecu(candidates, result, start, intermediate, target - candidates[start])
#            print intermediate
#            intermediate.pop()
#            start += 1
            
class Solution2:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def DFS(self, candidates,result, target, start, valuelist):
        length = len(candidates)
        if target == 0:
            return result.append(valuelist[:])
        for i in range(start, length):
            
            if target < candidates[i]:
                return
            
            self.DFS(candidates,result, target - candidates[i], i, valuelist + [candidates[i]])
        
    def combinationSum(self, candidates, target):
        candidates.sort()
        result = []
        self.DFS(candidates, result,target, 0, [])
        return result            
            
Obj = Solution()
print Obj.combinationSum([2,3,6,7],7)