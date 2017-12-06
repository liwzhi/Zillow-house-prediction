# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:34:07 2015

@author: weizhi
"""
#Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.
#
#Note:
#You must not modify the array (assume the array is read only).
#You must use only constant, O(1) extra space.
#Your runtime complexity should be less than O(n2).
#There is only one duplicate number in the array, but it could be repeated more than once.
#Credits:
#Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

class Solution():
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
    
        slow = nums[0]
        fast = nums[nums[0]]
        while (slow != fast):
            slow = nums[slow]
            fast = nums[nums[fast]]
        print slow
    
        fast = 0;
        while (fast != slow):
            fast = nums[fast]
            slow = nums[slow]
        return slow
obj = Solution()
print obj.findDuplicate([1,2,3,1,1,2,3,4,1,4,1])
        
def findDuplicate(nums):
    n = len(nums)
   # print n
    tortoise = nums[n-1]
    hair = nums[nums[n-1]-1]# 3
    if nums[n-1]==n:
        return n
  #  print tortoise
  #  print hair
    while tortoise!=hair:
      #  print tortoise  # 5
      #  print hair   # 3
        tortoise = nums[tortoise-1]  # 2, 3
        hair = nums[nums[hair-1]-1] # 3

    tortoise = n
    while hair!=tortoise:
  
        tortoise = nums[tortoise-1]  #2
        hair = nums[hair-1] # 
    return tortoise
nums = [1,3,4,2,2]
print findDuplicate(nums)
