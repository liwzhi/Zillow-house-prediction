# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:07:51 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:35:22 2015

@author: weizhi
"""

#%%
class Solution:
    # @return a list of lists of integers
    def combine(self, n, k):
        result = []
        self.combineRecu(n, result, 0, [], k)
        return result
    
    def combineRecu(self, n, result, start, intermediate, k):
        if k == 0:
            result.append(intermediate[:])
        for i in xrange(start, n):
            intermediate.append(i + 1)
            print intermediate
            self.combineRecu(n, result, i + 1, intermediate, k - 1)
            intermediate.pop()
      #%%      
    def permute(self,n):
        result = []
        flag = [False]*len(n)
        self.permutationRecur(n,result,[],flag)
        return result
    def permutationRecur(self,n,result,stk,flag):
        if len(stk)==len(n):
            result.append(stk[:])
            return result
        for i in range(len(n)):
            item = n[i]
            if not flag[i]:
                flag[i]=True
                stk.append(item)
                self.permutationRecur(n,result,stk,flag)
                stk.pop()
                flag[i] = False
                
                #%%
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        self.recur(nums,result,[],[False]*len(nums))
        return result
    def recur(self,nums,result,stk,flag):
        if len(stk)==len(nums):
            if stk[:] not in result:
                result.append(stk[:])
        for i in range(len(nums)):
            item = nums[i]
            if not flag[i]:
                flag[i] = True
                stk.append(item)
                self.recur(nums,result,stk,flag)
                stk.pop()
                flag[i] = False
                
    # @param num, a list of integer
    # @return a list of lists of integers
    def permuteUnique2(self, num):
        length = len(num)
        if length == 0: return []
        if length == 1: return [num]
        num.sort()
        res = []
        previousNum = None
        for i in range(length):
            if num[i] == previousNum: continue
            previousNum = num[i]
            for j in self.permuteUnique(num[:i] + num[i+1:]):
                res.append([num[i]] + j)
        return res        
                
obj = Solution()
#print obj.combine(4,2)

print obj.permute(range(3))
print obj.permuteUnique([0,1,2])