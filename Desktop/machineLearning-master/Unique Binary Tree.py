# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 23:28:37 2015

@author: weizhi
"""
# Catalan number
class Solution(object):
    def numTrees(self,n):
        dp = [0 for i in range(3)]
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        if n<=2:
            return dp[n]
        else:
            dp +=[0 for i in range(3,n+1)]
            for i in range(3,n+1):
                for j in range(1,i+1):
                    dp[i] += dp[j-1]*dp[i-j]
        return dp[n]