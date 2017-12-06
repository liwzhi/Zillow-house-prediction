# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:01:03 2015

@author: weizhi
"""

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if len(preorder)==0 or len(inorder)==0:
            return None
        curr = preorder.pop(0)
        root = TreeNode(curr)
        mid = inorder.index(curr)
        root.left = self.buildTree(preorder,inorder[:mid])
        root.right = self.buildTree(preorder,inorder[mid+1:])
        return root
        
    def buildTree2(self,inorder,postorder):
        if len(inorder) ==0 or len(postorder) ==0:
            return None
        curr = postorder.pop(-1)
        root = TreeNode(curr)
        mid = inorder.index(curr)
        root.right = self.buildTree2(inorder[mid+1:],postorder)
        root.left = self.buildTree2(inorder[:mid],postorder)
        return root

obj = Solution()
obj.buildTree([1,2,3],[2,1,3])