# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:44:06 2015

@author: weizhi
"""





class Solution(object):
    def generateParenthesis(self,n):
        res = []
        if (n<=0):
            return res
        self.helper(n,n,'',res)
        return res
        
    def helper(self,l,r,item,res):
        if(r<l):
            return
        if (l==0 and r ==0):
            res.append(item)
        if l>0:
            self.helper(l-1,r,item+'(',res)
        if r>0:
            self.helper(l,r-1,item+')',res)
aa = Solution()
print aa.gene(3)