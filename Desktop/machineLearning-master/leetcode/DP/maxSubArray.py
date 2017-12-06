# -*- coding: utf-8 -*-

class Solution(object):
    def maxSubArray(self, nums):
        if len(nums)<=0:
            return 0
        dp = [0]*len(nums)
        dp[0] = nums[0]
        local = dp[0]
        for i in range(1,len(nums)):
            if local<0:
                dp[i] = max(local,nums[i])
            else:
                dp[i] = local + nums[i]
            local = dp[i]
     #       print local
        return max(dp)

aa = Solution()
print(aa.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))