# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:18:16 2015

@author: weizhi
"""
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        mapValue = {}
        pre = None
        maxLen = 0
        for i in range(len(s)):
            item = s[i]
            if item not in mapValue.keys():
                mapValue[item] = (pre,i)
                pre = item
                maxLen = max(maxLen,len(mapValue.values()))
            
            else:
                currItem = item
                preDel = mapValue[item]
                while item:
                    preDel = mapValue[item]
                    del mapValue[item] 
                    if preDel==None:
                        item = None
                    else:
                        item = preDel[0]
                while mapValue:
                    index = preDel[1]
                    nextKey =s[index]
                    mapValue[nextKey] = None
                
                        
                mapValue[currItem] = pre
        return maxLen
                    
obj = Solution()
print obj.lengthOfLongestSubstring('bbbb')