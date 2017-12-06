# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:17:59 2015

@author: weizhi
"""
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if not nums or not s:
            return 0
        toCheck = set([s])
        depth = 0
        Dict = {}
        while toCheck:
            nextLevel = []
            count = 0
            keys = range(len(nums)-depth+1)
            for item in toCheck:
                for index in range(depth,len(nums)):
                  #  depth +=1
                    candidate = nums[index-depth]
                    
                    key = keys[index]

                    if key not in Dict:
                        Dict[key] = [item]
                    if candidate>item:
                        break
                    nextLevel.append(item-candidate)
                    Dict[key].append(item-candidate)
                    if 0 in Dict[key]:
                        return len(Dict[key])
                    print Dict
            depth+=1
            toCheck = nextLevel
        return 0
obj = Solution()
print obj.minSubArrayLen(7,[2,3,1,2,4,3])
    
            
                    
                
