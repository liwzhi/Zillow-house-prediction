# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:00:49 2015

@author: weizhi
"""


import heapq
class UndirectedGraphNode(object):
    def __init__(self,x):
        self.label = x
        self.neighbors = []
class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class ListNode(object):
    def __init__(self,x):
        self.val = x
        self.next = None
class Solution:
    # @return an integer
    def minDistance(self,word1,word2):
        len1 = len(word1)
        len2 = len(words2)
        dp = [[0 for i in range(len2+1)] for j in range(len1+1)]
        
        for i in range(len1+1):
            dp[i][0] = 1
        for j in range(len2+1):
            dp[0][j] = 1
        
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                if word1[i-1] ==word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) +1 # dp[i-1][j], i-1 match j, i-1,j-1 replace
        return dp[len1][len2]
    
    def isBalanced(self,root):
        return (self.depth(root)>=0)
    
    
    def depth(self,root):
        if root==None:
            return 0
        left = self.depth(root.left)
        right = self.depth(root.right)
        if abs(left-right)>1 or left<0 or right<0:
            return -1
        return max(left,right) +1
    
    def cloneGraph(self, node):
        if node ==None:
            return 
        stk = [node]
        while stk:
            nextLevel = []
            for item in stk:
                newNode = UndirectedGraphNode(item.label)
                if item ==node:
                    result = newNode
                neighborList = []
                for neighbor in item.neighbors:
                    neighborList.append(neighbor)
                    nextLevel.append(neighbor)
                newNode.neighbors = neighborList
            stk = nextLevel
        return result
    
    def preorder(self,root):
        stk = []
        result = []
        while root!=None or stk:
            if root!=None:
                stk.append(root)
                result.append(root.val)
                root = root.left
            else:
                root = stk.pop()
                root = root.right
        return result
    
    def wordBreak(self,s,wordDict):
        if s==None:
            return None
        flag = [False for i in range(len(s)+1)]
        for i in range(len(s)+1):
            st1 = s[:i]
            if str1 in wordDict:
                flag[i] = True
            j = i
            if flag[i]:
                while j <=len(s):
                    str2 = s[i:j]
                    if str2 in wordDict:
                        flag[j] = True
                    j+=1
            return flag[len(s)]
        def maxSubArray(self,nums):
            if not nums:
                return 
            local = 0
            maxSum = 0
            for item in nums:
                local = max(local,item+local)
                maxSum = max(maxSum,local)
            return maxSum
        
        
        def largestNumber(self,nums):
            if not nums:
                return ''
            Dict = {}
        
        
        def mergeKLists(self,lists):
            h = []
            preNode = listNode(0)
            head = preNode
            
            for node in lists:
                if node:
                    heapq.heappush(h,(node.val,node))
            while h:
                root = heapq.heappop(h)[1]
                head.next = root
                
                if root.next:
                    heapq.heappush(h,(root.next.val,root.next))
                head = head.next
            return preNode.next
                
            
            
                        
                        
                        
                        
                        
                
                
                
                
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
    