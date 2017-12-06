# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:46:14 2015

@author: weizhi
"""

class Solution(object):
    def permute(self,num):
        if len(num)==0:
            return []
        if len(num)==1:
            return [num]
        result = []
        flag = [False for i in range(len(num))]
        self.helper(num,result,flag,[])
        return result
        
    def helper(self,num,result,flag,stk):
        if len(stk) == len(num):
            return result.append(stk[:]) 
        for i in range(len(num)):
            if not flag[i]:
                flag[i] = True
                stk.append(num[i])
               # print stk
                self.helper(num,result,flag,stk)
                stk.pop()
                flag[i] = False
Obj = Solution()
result = Obj.permute([i+1 for i in range(4)])
print result
                
                
                