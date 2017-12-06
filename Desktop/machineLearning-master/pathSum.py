# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:10:43 2015

@author: weizhi
"""
class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def sumNumbers(self,root):
        if root ==None:
            return 
        result = []
        stk = root.val
        result = self.helper(root,result,stk)
        return sum(result)
    def helper(self,root,result,stk):
     #   print stk
        if root.left ==None and root.right==None:
            result.append(stk)
           # return result
        print stk
        if root.left:
            #stk = stk*10 + root.left.val
            self.helper(root.left,result,stk*10 + root.left.val)
        if root.right:
           # stk = stk*1 + root.right.val
            self.helper(root.right,result,stk*10 + root.right.val)
        return result 

a = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
a.left= b
a.right = c

obj = Solution()
result = obj.sumNumbers(a)
print result
        
        
        
        