# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:58:08 2015

@author: weizhi
"""

class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n<=0:
            return 0
        result = []
        pre = str(n)[0]
        count = 1
        i = 1
        flag = 0
        while(i<len(str(n))):
            item = str(n)[i]
            flag = 0
            count = 1
            while(item==pre and i<len(str(n))):
                count+=1
                pre = item
                item = str(n)[i]
                i+=1
                flag = 1
            if flag ==1:
                result.append(str(count)+str(item))
            if flag ==0:
                i+=1
                result.append(str(count) + str(pre))
                pre = item
        if flag ==1:
            return ''.join(result)
        if flag ==0:
            result.append(str(count)+str(pre))
            return ''.join(result)     
                
            
obj = Solution()
print obj.countAndSay(1371)