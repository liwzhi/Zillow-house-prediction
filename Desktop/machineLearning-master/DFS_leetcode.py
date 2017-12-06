# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:43:28 2015

@author: weizhi
"""
class ListNode():
    def __init__(self,x):
        self.val = x
        self.left =None
        self.right = None


class Solution(object):
    def maxDepth(self,root):
        if root ==None:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right)) +1
    def isValidBST(self,root):
        if root ==None:
            return True
        if root.left and root.right:
            if root.left.val>=root.val or root.val>=root.right.val:
                return False
            return self.isValidBST(root.left) and self.isValidBST(root.right)
        if root.left and root.right==None:
            if root.left.val>=root.val:
                return False
            else:
                return self.isValidBST(root.left)
        if root.right and root.left==None:
            if root.right.val<=root.val:
                return False
            else:
                return self.isValidBST(root.right)
        return True
    def isValidBST2(self,root):
        if root ==None:
            return True
        flag = 1
        flag = self.helper(root,flag)
        if flag ==0:
            return False
        else:
            return True

    def isSymmetric(self,root):
        if root ==None:
            return True
        lRoot = root.left
        rRoot = root.right
        return self.helper2(lRoot,rRoot)
    def helper2(self,lRoot,rRoot):
        if lRoot ==None and rRoot ==None:
            return True
        if lRoot ==None and rRoot!=None:
            return False
        if lRoot !=None and rRoot ==None:
            return False
        if lRoot.left!=rRoot.right or lRoot.right!=rRoot.left:
            return False
        return self.helper(lRoot.left,rRoot.right) and self.helper(lRoot.right,rRoot.left)

    def helper3(self,root,result,stk):
        """
        Given a binary tree, return all root-to-leaf paths.

        For example, given the following binary tree:
        
        """
        if root.left ==None and root.right==None:
            result.append(stk)
        if root.left !=None:
            self.helper3(root.left,result,stk+'->'+str(root.left.val))
        if root.right !=None:
            self.helper3(root.right,result,stk+'->' + str(root.right.val))
        return result
    def binaryTreePaths(self,root):
        if root ==None:
            return []
        result = []
        stk = str(root.val)
        result = self.helper3(root,result,stk)
        return result    
    
    
    def pathSum(self,root,sum):
        result = []
        stk = []
        if root ==None:
            return result
        result = self.helper4(root,result,sum,stk = [' '])
        return result
    
    
    def helper4(self,root,result,sum,stk):
        print stk
        if root.left ==None and root.right ==None:
            if sum == root.val:
                result.append(stk.append(root.val)[1:])
        if root.left:
            self.helper4(root.left,result,sum-root.val,stk.append(root.val))
        if root.right:
            self.helper4(root.right,result,sum-root.val,stk.append(root.val))
        return result
        

    # @param root, a tree node
    # @param sum, an integer
    # @return a list of lists of integers
    def pathSum2(self, root, sum):
        return self.pathSumRecu([], [], root, sum)

    
    def pathSumRecu(self, result, cur, root, sum):
        if root is None:
            return result
        
        if root.left is None and root.right is None and root.val == sum:
            result.append(cur + [root.val])
            return result
        
        cur.append(root.val)
        self.pathSumRecu(result, cur, root.left, sum - root.val)
        self.pathSumRecu(result, cur,root.right, sum - root.val)
        cur.pop()
        return result        
        


a = ListNode(1)
b = ListNode(6)
c = ListNode(2)    
a.left = b
a.right = c
        
Obj = Solution()
print Obj.pathSum(a,7)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            