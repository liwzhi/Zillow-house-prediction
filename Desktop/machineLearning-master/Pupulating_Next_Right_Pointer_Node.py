# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:44:32 2015

@author: weizhi
"""

class Solution(object):
    """
    for pupulating next right pointer node, 1, 2
    """
    def connect(self,root):
        if root ==None:
            return None
        stk = [root]
        while(stk):
            nextLevel = []
            for i in range(len(stk)):
                item = stk[i]
                if item.left:
                    nextLevel.append(item.left)
                if item.right:
                    nextLevel.append(item.right)
                if i<len(stk)-1:
                    stk[i].next = stk[i+1]
            stk = nextLevel