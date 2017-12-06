# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:31:26 2015

@author: weizhi
"""

class Solution(object):
    def quickSort(self,nums,p,r):
        
     #   p = 0
      #  r = len(nums) -1
        if p<r:
            q = self.partition(nums,p,r)
          #  print q
            if q==len(nums)/2:
                return nums[q]
            self.quickSort(nums,p,q-1)
            self.quickSort(nums,q+1,r)
       # return nums[q]
       # print nums
    def partition(self,A,p,r):
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
        
    def mergeSort(self,nums):
        if len(nums)<=1:
            return nums
        mid = len(nums)//2
        left = self.mergeSort(nums[:mid])
        right = self.mergeSort(nums[mid+1:])
        return self.merge(left,right)
    def merge(self,lefthalf,righthalf):
        i=0
        j=0
        k=0
        alist = [0 for ii in range(len(lefthalf)+len(righthalf)-1)]
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
        return alist
                
        
        
obj = Solution()
nums = [1,2,0,-1,300,8,10]
print obj.quickSort([1,2,0,-1,300,8,10],0,6)
print obj.mergeSort(nums)
                
                
        
        