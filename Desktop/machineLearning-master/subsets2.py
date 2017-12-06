# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:44:04 2015

@author: weizhi
"""

class Solution(object):
    def subsetsWithDup(self,nums):
        result = []
        
        self.helper(nums,result,0,[])
        return result
        
    def helper(self,nums,result,start,stk):
        result.append(stk[:])
        for i in range(start,len(nums)):
            stk.append(nums[i])
            print stk
            self.helper(nums,result,start+1,stk)
            stk.pop()
    
    def helper2(self,nums,index):
        if index<0:

            return [[]]
        result = self.helper2(nums,index-1)
        for i in range(len(res)):
            

obj = Solution() 
result = obj.subsetsWithDup([1,2,3])   
print result    
            
        
        