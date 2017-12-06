# -*- coding: utf-8 -*-

class Solution(object):
    def climbStairs(self, n):  
        if n<2:
            return n
        dp = [ 0 for i in range(n+1)]
        
        dp[1] =1
        dp[2] = 1
        for i in range(3,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
        