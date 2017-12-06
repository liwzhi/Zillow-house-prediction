# -*- coding: utf-8 -*-

class TreeNode(object):
    def __init__(self,x):
        self.right = None
        self.left = None
        self.val = x

class Solution:
    def binaryTreePaths(self,root):
        if root ==None:
            return []
        result = []
        self.helper(root,result,str(root.val))
        return result
    def helper(self,root,result,stk):
        if root.left==None and root.right ==None:
            result.append(stk)
        if root.left:
            self.helper(root.left,result,stk+"->" + str(root.left.val))
        if root.right:
            self.helper(root.right,result,stk +"->" + str(root.right.val))
aa = Solution()

root1 = TreeNode("3")
root1Left = TreeNode("2")

root1.left = root1Left

print(aa.binaryTreePaths(root1))
            
        
        
        
        