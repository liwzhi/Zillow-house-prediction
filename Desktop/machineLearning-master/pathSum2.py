# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:44:32 2015

@author: weizhi
"""
class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    """
    Wrong answer, need to check S
    """
    def pathSum2(self,root,sum):
     #   result = []
      #  self.helper(root,result,sum,[])
        return self.helper(root,[],sum,[])
    
    def helper(self,root,result,sum,stk):
        if root ==None:
            return result
        if root.left ==None and root.right==None and sum == root.val:
                stk.append(root.val)
                result.append(stk)
        #        print stk
         #       print result
                return result

        if root.left:
            stk.append(root.val)
            self.helper(root.left,result,sum-root.val,stk)
            stk.pop(-1)
        if root.right:
            stk.append(root.val)
            self.helper(root.right,result,sum-root.val,stk)
            stk.pop(-1)
        return result

a = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
d = TreeNode(4)
a.left = b
a.right = c
c.left = d

obj = Solution()
print obj.pathSum2(a,8)
        
