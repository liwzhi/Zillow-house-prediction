class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 
        if len(nums)==1:
            return 0
        if nums[1]<nums[0]:
            return 0
        pre = nums[0]
        for i in range(1,len(nums)-1):
            curr = nums[i]
            next = nums[i+1]
            if curr>pre and curr>next:
                return i
            else:
                curr = pre
        if nums[len(nums)-1]>nums[len(nums)-2]:
            return len(nums)-1
            