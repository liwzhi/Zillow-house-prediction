# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:33:08 2015

@author: weizhi
"""

class Solution(object):
    def permuteUnique(self,nums):
        if len(nums) ==0:
            return []       
        result = []
        nums.sort()
        used = [False for i in range(len(nums))]
        self.helper(nums,result,[],used)
        return result
    def helper(self,nums,result,stk,used):
        if len(nums) ==len(stk):
          #  print stk
           # print stk
            if stk[:] not in result:
                result.append(stk[:])
          #  print result
            return result
        for i in range(len(nums)):
            if i>0 and nums[i-1] == nums[i]:
               # print nums[i]
                pass
            if not used[i]:
                used[i] = True
                stk.append(nums[i])
                self.helper(nums,result,stk,used)
                stk.pop()
                used[i] = False
        return result
                
          #  stk = stk+[nums[i]]
            #self.helper(nums,result,stk + [nums[i]],start+1)
          #  stk.pop()
      #  return result
obj = Solution()
result = obj.permuteUnique([1,1,3,2])
print result

#print set(result)