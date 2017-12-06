# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:22:26 2015

@author: weizhi
"""

class UndirectedGraphNode(object):
    def __init__(self, x):
        self.label = x
        self.neighbors = []

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: UndirectedGraphNode
        :rtype: UndirectedGraphNode
        """
        if not node:
            return node
        stk = [node]
        while stk:
            nextLevel = []
            for item in stk:
                root = UndirectedGraphNode(item.label)
                if item==node:
                    result = root
             #   print root.label
                labs = []
                for neighbor in item.neighbors:
                    nextLevel.append(neighbor)
                    labs.append(neighbor)
                root.neighbors = labs
            stk = nextLevel
        return result
obj = Solution()
node = UndirectedGraphNode(1)
node1 = UndirectedGraphNode(1)
node2 = UndirectedGraphNode(2)
node3 = UndirectedGraphNode(3)

node.neighbors= [node1,node2,node3]
result =  obj.cloneGraph(node)