# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
       # result = []
        #self.recur2(root,result,0)
        #self.recur2(root,0)
        return self.recur2(root,0)
    def recur(self,root,result,sumPath):
        """
        Solution one
        """
        if root ==None:
            return result
        if root.left==None and root.right==None:
            result.append(sumPath*10 + root.val)
        self.recur(root.left,result,sumPath*10 + root.val)
        self.recur(root.right,result,sumPath*10 + root.val)
        return result
    def recur2(self,root,Sum):
        """
        Solution two 
        """
        if root ==None:
            return 0
        if root.left ==None and root.right==None:
            return Sum*10 + root.val
        left = self.recur2(root.left,Sum*10 + root.val)
        right = self.recur2(root.right,Sum*10 + root.val)
        return left + right
        