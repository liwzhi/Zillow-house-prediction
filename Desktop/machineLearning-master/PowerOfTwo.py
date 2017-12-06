# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:51:02 2015

@author: weizhi
"""

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:
            return False
        if n ==1:
            return True

        result = self.recur(abs(n),[])
        #print result
       # print result
        if not result:
            return False
        else:
            if result[0] ==0:
                return True
            else:
                return False

    def recur(self,n,result):
        if n<=2:
           # print result
            result.append(n%2)
            return result
        if n%2==0:
            self.recur(n/2,result)
        return result
        
         

obj = Solution()
print obj.isPowerOfTwo(128)
print obj.isPowerOfTwo(-16)