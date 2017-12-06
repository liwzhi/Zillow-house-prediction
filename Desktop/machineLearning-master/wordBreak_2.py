# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:12:14 2015

@author: weizhi
"""

class Solution():
    def wordBreak(self,s,wordDict):
        if s==None:
            return True
        Flag = [False for i in range(len(s)+1)]
        for i in range(len(s)+1):
            str1 = s[:i]
            if str1 in wordDict:
                Flag[i] = True
            j = i 
            if Flag[i]:
                while j<=len(s):
                    str2 = s[i:j]
                    if str2 in wordDict:
                        Flag[j] = True
                    j +=1
        return Flag[len(s)]
obj = Solution()
s = 'a'
wordDict= ['a']
s = 'leetcode'
wordDict = ['leet','code']
print obj.wordBreak(s,wordDict)