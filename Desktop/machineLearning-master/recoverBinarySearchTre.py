# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:44:06 2015

@author: weizhi
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        if root ==None:
            return 
        maxValue = float('infit')
        minValue = float('-infit')
    
    def findTheNodes(sef,root,maxValue,minValue,result):
#        if root ==None:
#           return result
        if root.val>=maxValue or root.val<=minValue:
            stk.append(root)
        #    result.append(stk)
            left = root.left
            right = root.right
            result.append((stk,left,right))
            count+=1
            if count ==2:
                root1 = result[0][0].pop()
                left1 = result[0][1].pop()
                right1 = result[0][2].pop()
                if result[0][0]!=None
                    root1Parent = result[0][0].pop()
                else:
                    root1Parent =None
                root2 = result[1][0].pop()
                left2 = result[1][1].pop()
                right2 = result[1][2].pop()
                if result[1][0] !=None:
                    root2Parent = result[1][0].pop()
                else:
                    root2Parent =None
                if root1Parent:
                    root1Parent
                
                
                
            return result
                
                
        
        if root.left!=None:
            self.findTheNodes(root.left,root.val,minValue,stk+[root],result)
        if root.right!=None:
            self.findTheNodes(root.right,maxValue,root.val,stk+[root],result)
        
                
        
        