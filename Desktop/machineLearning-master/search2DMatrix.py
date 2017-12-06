# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:16:47 2015

@author: weizhi
"""

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        row = len(matrix)
        col = len(matrix[0])
        low = 0
        high = row-1
        print col
        while low<=high:
            mid = low + (high-low)/2
            print mid
            if matrix[mid][0]==target:
                return True
            if matrix[mid][0]>target:
                high = mid-1
            else:
                low = mid+1
       # print matrix[:][0]
      #  print high
        if high<0:
            return False
        low1 = 0
        high1 = col-1
        
        while low1<=high1:
            mid2 = (high1+low1)/2
            if matrix[high][mid2]==target:
                return True
            if matrix[high][mid2]>target:
                high1 = mid-1
            else:
                low1 = mid +1
        return False
        
obj = Solution()
a = [
  [1,   2,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
#a = [[1]]
a= [[1,1]]
print obj.searchMatrix(a,3)