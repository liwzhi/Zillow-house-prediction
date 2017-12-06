# -*- coding: utf-8 -*-
class Solution(object):
    def permute(self,n):
        result = []
        flag = [False]*len(n)
        self.recur(n,result,[],flag)
        return result
    
    def recur(self,n,result,stk,flag):
        if len(n) == len(stk):
            result.append(stk[:])
            return result
        for i in range(len(n)):
            item = n[i]
            if not flag[i]:
                stk.append(item)
                flag[i] = True
                self.recur(n,result,stk,flag)
                stk.pop()
                flag[i] = False
        

