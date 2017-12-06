# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:29:56 2015

@author: weizhi
"""

class Solution():
    def wordBreak(self,s,wordDict):
        if s==None:
            return True
        res = [False]*(len(s)+1)
        res[0] = True
        for i in range(1,len(s)+1):
           # print s[:i+1]
           # print s[:i+1] in wordDict
            for j in range(i):
                if res[j] and s[j:i] in wordDict:
                    res[i] = True
        return res[len(s)]
obj = Solution()
print obj.wordBreak("a",['a'])