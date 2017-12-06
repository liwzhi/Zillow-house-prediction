# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:47:02 2015

@author: weizhi
"""

# Definition for a binary tree node.
class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution:
    # @param root, a tree node
    # @return nothing, do it in place
    def flatten(self, root):
        root = self.flattenRecu(root, None)
        return root
        
    def flattenRecu(self, root, list_head):
        if root != None:
            list_head = self.flattenRecu(root.right, list_head)
            list_head = self.flattenRecu(root.left, list_head)
            root.right = list_head
            root.left = None
           # print root.val
            if list_head!=None:
                print list_head.val
        
            
            return root
        else:
            return list_head

root = TreeNode(0)
left1 = TreeNode(1)
left2 = TreeNode(2)
right1 = TreeNode(3)
right2 = TreeNode(4)

root.left = left1
left1.left = left2
root.right = right1
right1.right = right2

obj = Solution()
root = obj.flatten(root)