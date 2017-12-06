# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:32:57 2015

@author: weizhi
"""

class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        dp = [[False for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)
        if len1+len2 !=len3:
            return False
        dp[0][0]=True
        for i in range(1,len1+1):
            if s1[i-1]==s3[i-1] and dp[i-1][0]:
                dp[i][0]= True
        for j in range(1,len2+1):
            if s2[j-1]==s3[j-1] and dp[0][j-1]:
                dp[0][j] = True
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                if (s1[i-1] == s3[i+j-1] and dp[i-1][j]==True) \
                 or(s2[j-1]== s3[i+j-1] and dp[i][j-1]==True):
                    dp[i][j]=True
        return dp[len1][len2]
obj = Solution()
print obj.isInterleave('a','','a')