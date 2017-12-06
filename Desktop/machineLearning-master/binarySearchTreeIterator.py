# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:34:45 2015

@author: weizhi
"""

# Definition for a  binary tree node
#Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

#Calling next() will return the next smallest number in the BST.

#Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.root = root
        self.stack = []
        self.helper(root)
    def helper(self,root):
        while root:
            self.stack.append(root)
            root = root.left

        

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) !=0

        
        

    def next(self):
        """
        :rtype: int
        """
        currRoot = self.stack.pop()
        self.helper(currRoot.right)
        return currRoot.val

# Definition for a  binary tree node
#Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

#Calling next() will return the next smallest number in the BST.

#Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator2(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.root = root
        self.result = self.traversal2(root,[])
        self.result = self.travseral(root)
        
        

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.result:
            return True
        return False
        
        

    def next(self):
        """
        :rtype: int
        """
        item = self.result[0]
       # print item
        self.result.remove(item)
        return item
    def travseral(self,root):
        result = []
        stk = []
        while root or stk:
            if root:
                stk.append(root)
                root = root.left
            else:
                root = stk.pop()
                result.append(root.val)
                root = root.right
        return result
    def traversal2(self,root,result):
        if root==None:
            return None
        self.traversal2(root.left,result)
        result.append(root.val)
        self.traversal2(root.right,result)
        return result
        
        

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())
        

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())