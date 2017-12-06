# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:51:27 2015

@author: weizhi
"""
class ListNode(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    """
    T(n) = 2*T(n/2) + O(1)  
    a = 2
    b = 2
    c = 0
    a > b**(c), so the time complexity is: O(2*logn) =O(logn)
    """
    def isSameTree(self,p,q):
        if p==None and q!=None:
            return False
        if p!=None and q==None:
            return False
        if p==None and q==None:
            return True
        if p.val != q.val:
            return False
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
        
        
a = ListNode(2)
a1 = ListNode(3)
a2 = ListNode(4)

b = ListNode(2)
b2 = ListNode(3)
b3 = ListNode(6)

a.left = a1
b.left = b2
a1.left = a2
b2.left = b3


obj = Solution()
print obj.isSameTree(a,b)
        
        
            