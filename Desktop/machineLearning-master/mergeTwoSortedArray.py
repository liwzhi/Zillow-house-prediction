# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:47:37 2015

@author: weizhi
"""

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """

        i = m-1
        j = n-1
        index = m+n-1
        zeros = [0]*n
        nums1 = nums1 + zeros
       # print nums1
        while i>=0 and j>=0:
            if nums1[i]>=nums2[j]:
                nums1[index] = nums1[i]
                i-=1
            else:
                nums1[index] = nums2[j]
                j-=1
              #  print j
            index-=1
        print nums1
        while j>=0:
            nums1[index] = nums2[j]
            index-=1
            j-=1
        print nums1
     #   return nums1
        
                
            
obj = Solution()
print obj.merge([0],0,[1],1)