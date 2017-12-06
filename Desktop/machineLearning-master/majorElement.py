# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 23:03:41 2015

@author: weizhi
"""

class Solution(object):
    def majorityElement(self,nums):
        dict = {}
        for n in nums:
            dict[n] = dict.get(n,0) +1
            if dict[n]>len(nums)/2:
                return n
b = Solution()
print b.majorityElement([1,2,3,4,1,1,1,1,1])


dict = {'Name': 'Zara', 'Age': 27}

print "Value : %s" %  dict.get('Age')
print "Value : %s" %  dict.get('Sex', "Never")

            
        