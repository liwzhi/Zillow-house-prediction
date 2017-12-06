# -*- coding: utf-8 -*-
class Solution(object):
    def isSymmetric(self,root):
        if root==None:
            return True
        return self.helper(root.left,root.right)
    
    def helper(self,q,p):
        if q==None and p ==None:
            return True
        if q!=None and p==None:
            return False
        if q==None and p!=None:
            return False
        if q.val !=p.val:
            return False
        return self.helper(q.left,p.right) and self.helper(q.right,p.left)