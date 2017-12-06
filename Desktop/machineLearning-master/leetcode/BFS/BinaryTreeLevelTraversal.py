# -*- coding: utf-8 -*-
class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.right = None
        self.left = None
    def levelOrder(self,root):
        result = []
        if root ==None:
            return result
        level = [root]
        while(level):
            nextLevel = []
            nextResult = []
            for item in level:
                nextResult.append(item.val)
                if item.left:
                    nextLevel.append(item.left)
                if item.right:
                    nextLevel.append(item.right)
            result.append(nextResult)
            level = nextLevel
        return result
