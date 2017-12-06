# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:04:26 2015

@author: weizhi
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root==None:
            return []
        stk = [root]
        result = []
        count = 1
        while stk:
            nextLevel = []
            value = []
            count+=1
            for vertex in stk:
                value.append(vertex.val)
                if vertex.left:
                    nextLevel.append(vertex.left)
                if vertex.right:
                    nextLevel.append(vertex.right)

            if count%2==1:
                # = value.reverse()
                value.reverse()
            result.append(value)
            stk = nextLevel
        return result
            
        