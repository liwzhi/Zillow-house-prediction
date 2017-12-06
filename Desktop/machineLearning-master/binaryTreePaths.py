# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:56:46 2015

@author: weizhi
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):

    def helper3(self,root,result,stk):
        if root.left ==None and root.right==None:
            result.append(stk)
        if root.left !=None:
            self.helper3(root.left,result,stk+'->'+str(root.left.val))
        if root.right !=None:
            self.helper3(root.right,result,stk+'->' + str(root.right.val))
        return result
    def binaryTreePaths(self,root):
        if root ==None:
            return []
        result = []
        stk = str(root.val)
        result = self.helper3(root,result,stk)
        return result      