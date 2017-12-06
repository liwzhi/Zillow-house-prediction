class Solution(object):
    """
    cannot change the order of array, Solution two works
    """
#    def moveZeroes(self, nums):
#        """
#        :type nums: List[int]
#        :rtype: void Do not return anything, modify nums in-place instead.
#        """
#        if len(nums)==0:
#            return 
#        index = len(nums)-1
#        i = 0
#        while(i<index):
#            curr = nums[i]
#            if curr ==0:
#                if nums[index]!=0:
#                    stk = nums[index]
#                    nums[index] = curr
#                    nums[i] = stk
#                    index-=1
#                if nums[index]==0:
#                    while (index>-1 and nums[index]==0):
#                        index-=1
#                    stk = nums[index]
#                    nums[index] = curr
#                    nums[i] = stk
#                    index-=1
#            i +=1
#        return nums
    def moveZeros(self,nums):
        if len(nums)==0:
            return 
        index = len(nums)-1
        i =0
        j =0
        while(i<=index and j<=index):
            if nums[i] ==0:
                j = i
                while(j<index and nums[j]==0):
                    j +=1
                middle = nums[j]
                nums[j] = nums[i]
                nums[i] = middle
            i +=1
   #     print nums

                
                       
        

obj = Solution()
print obj.moveZeros2([0,1,0,3,12,0])