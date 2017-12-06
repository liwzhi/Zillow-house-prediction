# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:23:32 2015

@author: weizhi
"""

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        result = [-1,-1]
        if not nums:
            return result
        left = 0
        right = len(nums)-1
        mid = (right-left)/2 + left
        while(left<=right):
            mid = (right-left)/2 + left
            value = nums[mid]
            if target>value:
                left = mid+1
            elif target<value:
                right = mid-1
            else:
                result[0] = mid
                result[1] = mid
                count = 0
               # print mid
                for i in range(mid-1,left-1,-1):
                  #  print mid-1
                  # print left-1 
                    while(i>=0 and target == nums[i]):
                        count +=1
                        i-=1
                        
                        
                    result[0] =result[0]-count
                    break
               # print count
                count =0
                for j in range(mid+1,right+1):
                    print j
                    while( j<len(nums) and target==nums[j]):
                        count +=1
                        j+=1
                    result[1] +=count
                    break
                return result
        return result
obj = Solution()
#print obj.searchRange([0,1,2,3,4,4,4,4,4,5,6,7],4)

print obj.searchRange([3,3,3],3)