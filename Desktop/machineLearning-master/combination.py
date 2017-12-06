# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:19:58 2015

@author: weizhi
"""

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
            self.combineRecu(n, result, i + 1, intermediate, k - 1)
            intermediate.pop()
            
class Solution2:
    def combine(self,n,k):
        result = []
        self.helper(n,result,0,[],k)
        return result
    def helper(self,n,result,start,intermediate,k):
        if k==0:
            result.append(intermediate[:])
        for i in range(start,n):
            intermediate.append(i+1)
            self.helper(n,result,i+1,intermediate,k-1)
            intermediate.pop()
class Solution4:
    def combine(self,nums):
        result = []
        self.helper(nums,result,0,[])
        return result
        
    def helper(self,nums,result,start,intermediate):
        result.append(intermediate)
        for i in range(start,len(nums)):
            intermediate.append(nums[i])
            self.helper(nums,result,i+1,intermediate)
            intermediate.pop()
        
        

if __name__ == "__main__":
    result = Solution2().combine(4, 2)
    print result
    result = Solution4().combine([2,3,4,5])
    print result
    
    
    
    
    
    
    
    