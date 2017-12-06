# -*- coding: utf-8 -*-
class Solution(object):
    def combine(self,n,k):
        result = []
        if n==0 or k ==0:
            return result
        return self.helper(n,result,[],k,0)
    def helper(self, n, result, stk, k, start):
        if len(stk) ==k:
            result.append(stk[:])
       #     print(stk[:])
            return result
        for i in range(start, n):
            item = i+1
            stk.append(item)
            self.helper(n,result,stk,k, i+1)
            stk.pop()
        return result

aa = Solution()
aa.combine(4,2)
        
        
        
        

