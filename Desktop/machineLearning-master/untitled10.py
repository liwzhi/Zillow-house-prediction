# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        leftMin = float("-inf")
        rightMax = float("inf")
        return self.recur(root,leftMin,rightMax)
    def recur(self,root,leftMin,rightMax):
        if root ==None:
            return True
        if root.val<=leftMin or root.val>=rightMax:
            return False
        return self.recur(root.left,leftMin,root.val) and self.recur(root.right,root.val,rightMax)