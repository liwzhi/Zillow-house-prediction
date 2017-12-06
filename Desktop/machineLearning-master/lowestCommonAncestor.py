# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 23:07:01 2015

@author: weizhi
"""

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root ==None:
            return 
        stk = [root]
        parent = {root:None}
        while p not in parent or q not in parent:
            node = stk.pop()
            if node.left:
                parent[node.left] = node
                stk.append(node.left)
            if node.right:
                parent[node.right] = node
                stk.append(node.right)
        ancestor = set()
        while p:
            ancestor.add(p)
            p = parent[p]
        while q not in ancestor:
            q = parent[q]
        return q

     