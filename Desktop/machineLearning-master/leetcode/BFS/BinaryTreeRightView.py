# -*- coding: utf-8 -*-
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def rightSideView(self,root):
        if root ==None:
            return []
        level = [root]
        result = [root.val]
        while(level):
            nextLevel = []
            for item in level:
                if item.right:
                    nextLevel.append(item.right)
                if item.left:
                    nextLevel.append(item.left)
            if nextLevel:
                result.append(nextLevel[0].val)
            level = nextLevel
        return result
            
            

                
            
