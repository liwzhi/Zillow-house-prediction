# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:38:09 2015

@author: weizhi
"""

class Solution(object):
    sign  = False
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        if not s:
            return []
        result = []
        self.helper(s,result,'',wordDict)
        return self.sign

                
    def helper(self,s,result,stk,wordDict):
        if len(s)==0:
            result.append(stk[1:])
            self.sign = True
            return self.sign
        for j in range(1,len(s)+1):
            word = s[:j]
            if word in wordDict:
                if not self.sign:
                    self.helper(s[j:],result,stk+' '+word,wordDict)
obj = Solution()
s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
Dict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
print obj.wordBreak(s,Dict)
        