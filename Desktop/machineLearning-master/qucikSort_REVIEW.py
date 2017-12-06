# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:54:08 2015

@author: weizhi
"""

class Solution():
    def quickSort(self,nums,left,right):
     #   left = 0
      #  right = len(nums)-1
        if left<right:
            p = self.partition(nums,left,right)
            print nums[p]
            self.quickSort(nums,left,p-1)
            self.quickSort(nums,p+1,right)
        return nums
    def partition(self,nums,p,r):
        x = nums[r]
        i = p-1
        for j in range(p,r-1):
            if nums[j]<=x:
                i+=1
                temp = nums[i]
                nums[i] = nums[j]
                nums[j] = temp
        temp = nums[i+1]
        nums[i+1] = x
        nums[r] = temp
        return i+1
    def quickSort2(self,nums,p,r):
        
     #   p = 0
      #  r = len(nums) -1
        if p<r:
            q = self.partition2(nums,p,r)
          #  print q
            if q==len(nums)/2:
                return nums[q]
            self.quickSort(nums,p,q-1)
            self.quickSort(nums,q+1,r)
        return nums
       # print nums
    def partition2(self,A,p,r):
        x = A[r]
        i = p-1
        for j in range(p,r):
            if A[j]<=x:
                i +=1
                curr = A[j]
                A[j] = A[i]
                A[i] = curr
        curr = A[i+1]
        A[i+1] = A[r]
        A[r] = curr
        return i+1        
        
        
obj = Solution()
print obj.quickSort([1,2,0,-1,300,8,10],0,6)
        