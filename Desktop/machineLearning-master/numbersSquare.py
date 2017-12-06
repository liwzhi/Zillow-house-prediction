# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:24:09 2015

@author: weizhi
"""

class Solution(object):
    sign = True
    result = 0
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        numsSet = []
        i = 1
        while i*i<=n:
            numsSet.append(i*i)
            i +=1
        self.result = n
        self.helper(numsSet,self.result,[],n,0)
        return self.result
    def helper(self,numsSet,result,stk,target,depth):
        if sum(stk)==target:
            self.result = min(self.result,len(stk))
            print stk
            return self.result
        if sum(stk)<target and depth<self.result-1:
            for item in reversed(numsSet):
                stk.append(item)
                self.helper(numsSet,result,stk,target,depth+1)
                stk.pop()
    def numsSquares(self,n):
        lst = []
        i = 1
        while i*i<=n:
            lst.append(i*i)
            i+=1
        cnt = 0
        toCheck = set([n])
        while 0 not in toCheck:
            cnt+=1
            temp = set()
            for x in toCheck:
                for y in lst:
                    if x<y:
                        break
                    temp.append(x-y)
            toCheck = temp
        return cnt
    
Obj = Solution()
print Obj.numSquares(9971)    
    
    
    
    
                

