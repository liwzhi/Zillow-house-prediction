class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums)<2:
            return nums
        index = len(nums)-1
        i = 0
        while(i<=index):
            curr = nums[i]
            if curr ==0:
                while(nums[index]==0 and index>-1):
                    index-=1
                stk = nums[index]
                nums[index] = curr
                curr = stk
            index -=1
            i+=1
        return nums
              #  self.swap(curr,nums[index])
obj = Solution()
print obj.moveZeroes([1,3,12,0,0])