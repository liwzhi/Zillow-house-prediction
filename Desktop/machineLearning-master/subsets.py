# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:10:36 2015

@author: weizhi
"""

class Solution(object):
    def subsets(self,S):
        S.sort()
        solution = []
        self.Help(S,0,[],solution)
        return solution
        
    def Help(self,S,index,tempSolution,solution):
        if index ==len(S):
           # print tempSolution
            solution.append(list(tempSolution))
        else:
            self.Help(S,index+1,tempSolution,solution)
            tempSolution.append(S[index])
            self.Help(S,index+1,tempSolution,solution)
            tempSolution.pop()

Obj = Solution()
print Obj.subsets([1,3,4])
        