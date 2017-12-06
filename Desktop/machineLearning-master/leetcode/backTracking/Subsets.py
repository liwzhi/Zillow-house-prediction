# -*- coding: utf-8 -*-
class Solution:
    def subsets(self,nums):
        result = []
        if len(nums)==0:
            return result
        return self.helper(sorted(nums),[],[],0)
    def helper(self,nums,result,stk,index):
        if len(nums) ==index:
            result.append(stk[:])
            print("another item")
            print stk[:]
            return result
        item = nums[index]
        self.helper(nums,result,stk,index+1)
        self.helper(nums,result,stk + [item], index+1)
        return result

a = Solution()
a.subsets([1,2,3])
  
