# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:46:05 2015

@author: weizhi
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param {TreeNode} root
    # @return {integer}
    result = float('inf')
    def minDepth(self, root):
        if not root:
            return 0
        self.helper(root,1)
        return self.result
    
    def helper(self,root,depth):
        if root.left ==None and root.right==None:
            self.result = min(self.result,depth)
        if root.left and depth<self.result:
            self.helper(root.left,depth+1)
        if root.right and depth<self.result:
            self.helper(root.right,depth+1)
    

    
    
    
    
    
    
    
    
    
    
    
    
