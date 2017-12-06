# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:15:19 2015

@author: weizhi
"""

class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right =None

class Solution(object):
    """
    Need more thinking, wrong answers
    """
    def recoverTree(self,root):
        if root ==None:
            return 
        
        Max = 2**(31)
        Min = -2**(31)+1
        result = []
        self.helper(None,root,Max,Min,result,0)
        return root
    def helper(self,rootPre,root,Max,Min,result,flag):
        if root ==None:
            return None
        if root.val>Max:
            result.append([rootPre,root,root.left,root.right,'left'])
            flag +=1
        if root.val<Min:
            result.append([rootPre,root,root.left,root.right,'right'])
            flag +=1
        if flag ==2:
            newNode = TreeNode(0)
            newNode = result[0][1]
            if result[0][4] == 'left':
                result[0][0].left = result[1][1]
                result[1][1].left = result[0][2]
                result[1][1].right = result[0][3]
            if result[0][4] =='right':
                result[0][0].right = result[1][1]
                result[1][1].left = result[0][2]
                result[1][1].right = result[0][3]
                

            if result[1][4] == 'left':
                result[1][0].left = newNode
                newNode.left = result[1][2]
                newNode.right = result[1][3]
            if result[0][4] =='right':
                result[1][0].right = newNode
                newNode.left = result[1][2]
                newNode.right = result[1][3]
        
        return self.helper(root,root.left,root.val,Min) and \
                self.helper(root,root.right,Max,root.val)
        