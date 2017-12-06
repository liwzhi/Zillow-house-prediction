# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:01:05 2015

@author: weizhi
"""

class Solution(object):
    # wrong solution, need more thinking in this part, using the flag parts
    def combinationSum2(self,candidates,target):
        result = []
        flag = [False for i in range(len(candidates))]
        self.helper(candidates,result,0,[],target,flag)
        
        return result
    def helper(self,candidates,result,start,stk,target,flag):
        if target<0:
            return 
        if target==0:
            result.append(stk[:])
        for i in range(start,len(candidates)):
            curr = flag[i]
            while not curr:
                print flag
                flag[i] = True
                self.helper(candidates,result,i,stk+[candidates[i]],target-candidates[i],flag)
         #   flag[i] = False 
class Solution2(object):
    def combinationSum2(self,candidates,target):
        result = []
       # flag = [False for i in range(len(candidates))]
        self.helper(sorted(candidates),result,0,[],target)
        
        return result
    def helper(self,candidates,result,start,stk,target):
        if target<0:
            return 
        if target==0:
            result.append(stk[:])
        for i in range(start,len(candidates)):
            if i>start and candidates[i]==candidates[i-1]:
                continue
            #if candidates[i] == candidates[i+1]:
             #   continue
          #  curr = flag[i]
         #   while not curr:
               # flag[i] = True
            self.helper(candidates,result,i+1,stk+[candidates[i]],target-candidates[i])
              #  flag[i] = False             
Obj = Solution()
result = Obj.combinationSum2([10,1,2,7,6,1,5],8)
print result