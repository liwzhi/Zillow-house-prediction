# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:03:27 2015

@author: weizhi
"""

class Solution(object):
    def minPathSum(self,grid):
        col = len(grid[0])
        row = len(grid)
        dp = [[0 for i in range(col)] for j in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1,col):
            dp[0][i] = grid[0][i] + dp[0][i-1]
        for j in range(1,row):
            dp[j][0] = grid[j][0] + dp[j-1][0]
        for i in range(1,row):
            for j in range(1,col):
                dp[i][j] = grid[i][j] + min(dp[i-1][j] , dp[i][j-1])
        return dp[row-1][col-1]
        