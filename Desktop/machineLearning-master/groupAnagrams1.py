# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:33:36 2015

@author: weizhi
"""

class Solution(object):
    def groupAnagrams(self, strs):
        if not strs:
            return 
        Dict = {}
        for string in strs:
            key =''.join(sorted(string))
            if key in Dict:
                Dict[key].append(string)
            else:
                Dict[key] = [string]
        result = []
        for row in Dict.values():
            row = sorted(row)
            result.append(row)
        return result
obj = Solution()
strings = ["eat", "tea", "tan", "ate", "nat", "bat"]
result = obj.groupAnagrams(strings)    
print result        