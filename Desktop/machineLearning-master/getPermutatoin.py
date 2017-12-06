# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:59:00 2015

@author: weizhi
"""

'''
Time Limit Exceeded
need more thinking
'''

class Solution(object):
   # result = 0
    count = 0
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        result = []
        stk = []
        flag = [False for i in range(n)]
        #count = 0
        result = self.recur(n,stk,k,flag,result)
        return result
    def recur(self,n,stk,k,flag,result):
        #print stk
        if len(stk) == n:
            self.count +=1
            #print count
          #  print self.count
           # print k
            if self.count == k:
              #  print result
                result.append(stk)
                print result
                
                
                return result
        for i in range(n):
           # stk = stk + [i]
            #flag[i-1] = True
            if not flag[i]:
                stk = stk + [i+1]
                flag[i] = True
                self.recur(n,stk,k,flag,result)
                stk.pop()
                flag[i] = False
        return result
        
        
        
class Solution2(object):
    count = 0
    def permute(self,nums,k):
        if nums ==0:
            return []       
        result = []
        used = [False for i in range(nums)]
        result = self.helper(nums,result,[],used,k)
        print result
        return ''.join([str(item) for item in result[0]])
    def helper(self,nums,result,stk,used,k):
        if nums ==len(stk):
            self.count +=1
            if self.count == k:
                result.append(stk[:])
          #  print result
            return result
        for i in range(nums):
            if not used[i]:
                used[i] = True
                stk.append(i+1)
                self.helper(nums,result,stk,used,k)
                stk.pop()
                used[i] = False
        return result        
            
obj = Solution2()
print obj.permute(9,54494)  


obj = Solution()
#print obj.getPermutation(1,1)      