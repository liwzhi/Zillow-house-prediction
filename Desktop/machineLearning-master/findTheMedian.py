# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:29:02 2015

@author: weizhi
"""

class Solution():
    def findMid(self,nums,mid):
        l = 0
        r = len(nums)-1
        while l<r:
            p = self.partition(nums,l,r,mid)
            if p==mid:
                return nums[p]
            if p>mid:
                self.findMid(nums[:p],mid-(r-p+1))
            if p<mid:
                self.findMid(nums[p:],mid-(p+1))
        return 
    def partition(self,nums,p,r,mid):
        x = nums[r]
        i = p-1
        for j in range(p,r-1):
            if nums[j]<=x:
                i+=1
                temp = nums[i]
                nums[i] = nums[j]
                nums[j] = temp
        temp= nums[i+1]
        nums[i+1] = x
        nums[r] = temp
        return i+1
obj = Solution()
a = [1,2,3,4,510,1,2,3,-1,4]
obj.findMid(a,len(a)//2)
                
                
            
            
            
            
            
            
            
            
            
            
            
            
            