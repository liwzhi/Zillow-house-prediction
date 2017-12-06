# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:59:17 2015

@author: weizhi
"""
class TreeNode():
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    sign = True
    commonRoot = TreeNode(0)
    def subTree(self,root1,root2):
        self.searchTree(root1,root2)
        root = self.commonRoot
        h1 = self.getHeight(root)
        h2 = self.getHeight(root2)
        
        if h1!=h2:
            return False
        return self.checkSameTree(root,root2)

    
    def getHeight(self,root):
        if root ==None:
            return 0
        left = self.getHeight(root.left)
        right = self.getHeight(root.right)
        return max(left,right)+1
    def searchTree(self,root1,root2):
        if root1.val == root2.val:
            self.sign = False
            self.commonRoot =root1
        if root1.left and self.sign:
            self.searchTree(root1.left)
        if root2.right and self.sign:
            self.searchTree(root2.right)
    def checkSameTree(self,root1,root2):
        if root1==None and root2 ==None:
            return True
        if root1 !=None and root2==None:
            return False
        if root1==None and root2!=None:
            return False
        if root1.val!=root2.val:
            return False
        return self.checkSameTree(root1.left,root2.left) and self.checkSameTree(root1.right,root2.right)
            
            
            
            
            
            
            
            
            