# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# BFS solution
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root ==None:
            return 0
        
        
        stk = [root]
        depth = 0
        while stk:
            nextLevel = []
            depth+=1
            for item in stk:
                if item.left:
                    nextLevel.append(item.left)
                if item.right:
                    nextLevel.append(item.right)
            stk = nextLevel
        return depth
class Solution2(object):
    result = float('-inf')
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root ==None:
            return 0
        self.helper(root,1)
        return self.result
    def helper(self,root,depth):
        if root.left==None and root.right ==None:
            self.result = max(self.result,depth)
        if root.left:
            self.helper(root.left,depth+1)
        if root.right:
            self.helper(root.right,depth+1)                