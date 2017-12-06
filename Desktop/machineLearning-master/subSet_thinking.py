# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:24:50 2015

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
            print tempSolution
        else:
            self.Help(S,index+1,tempSolution,solution)
            tempSolution.append(S[index])
            self.Help(S,index+1,tempSolution,solution)
            tempSolution.pop()

class Solution2(object):
    def subsets(self,s):
        if len(s)==0:
            return [[]]
       # stk = []
        result =[[]]
        for i in range(len(s)):
            size = len(result)
            for j in range(size):
                item = result[j]
                item.append(s[i])
              #  print item
                item.append(item)
                
                result.append(item)
        return result
obj = Solution2()
result = obj.subsets([1,2,3])
print result




