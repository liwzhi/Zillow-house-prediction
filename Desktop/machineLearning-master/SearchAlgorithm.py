# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:32:22 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:22:09 2015

@author: weizhi
"""

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

def dfs(graph,start):
    visited = set()
    stack = [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex]-visited)
    return visited

class ListNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
    

def maxDepth(root):
    if root==None:
        return 0
    return max(maxDepth(root.left),maxDepth(root.right)) +1
    
class Solution:
    def isSameTree(self,p,q):
        if p==None and q==None:
            return True
        if p!=None and q==None:
            return False
        if p==None and q!=None:
            return False
        if p.val!=q.val:
            return False
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
    def connect(self,root):
        if root==None:
            return
        curr = [root]
        while(curr):
            level = []
            for index in range(len(curr)):
                item = curr[index]
                if item.left:
                    level.append(item.left)
                if item.right:
                    level.append(item.left)
                if index<len(curr)-1:
                    item.next = item[index+1]
            curr = level
    def sortedToArrayToBST(self,num):
        if len(num)==0:
            return None
        mid = len(num)/2
        root = ListNode(num[mid])
        root.left = self.sortedToArrayToBST(num[:mid])
        root.right = self.sortedToArrayToBST(num[mid+1:])
        return root
    def balancedTree(self,root):
        if root==None:
            return True
        if abs(self.balancedTreeHelper(self.left)-self.balancedTreeHelper(self.right))>1:
            return False
        return self.balancedTreeHelper(root.left) and self.balancedTreeHelper(root.right)
    
    def balancedTreeHelper(self,root):
        if root==None:
            return 0
        if root.left==None and root.right ==None:
            return 1
        if root.left==None:
            return self.balancedTreeHelper(root.right) +1
        if root.right==None:
            return self.balancedTreeHelper(root.left) +1
        return max(self.balancedTreeHelper(root.left),self.balancedTreeHelper(root.right)) +1
    
        
        
        

            
        
    
print dfs(graph,'A')
            
def dfs2(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for next in graph[start] - visited:
        dfs2(graph, next, visited)
    return visited
print dfs2(graph,'A',visited=None)

