# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:32:12 2015

@author: weizhi
"""

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        result = []
        if not s:
            return result
        
        result = []
        for i in range(len(s)+1):
            str1 = s[:i]
            stk = []
            parent = {}
            flag = [False for ii in range(len(s)+1)]
            if str1 in wordDict:
                flag[i] = True

                parent[i] = 0
                i = len(str1) 
                j = len(str1) 
                pre = len(str1) 


                while j<=len(s):
                    str2 = s[pre:j]
                    if str2 in wordDict:
                        flag[j] = True
                        parent[j] = pre
                    if flag[len(s)]:
                        p = j
                        while p!=0:
                            print p
                            stk.append(s[parent[p]:p])
                            p = parent[p]
                        result.append(stk)
                    pre = j
                    j+=1
                    
        return result
obj = Solution()

s = "catsanddog"
dict = ["cat", "cats", "and", "sand", "dog"]
print obj.wordBreak(s,dict)
                        
                        
                        
                        
                        
                        
                        
                        
                        
            
 