# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:59:53 2015

@author: weizhi
"""

import heapq
 

a = [3,4,5,6,709,88.90]
h = []
for item in a:
    heapq.heappush(h,(item,0))
result = []
while h:
    result.append(heapq.heappop(h))
print result



# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        h = []
        
        head = ListNode(0)
        root = head
        for node in lists:
            if node:
                heapq.heappush(h,(node.val,node))
        while h:
            item = heapq.heappop(h)
            root.next= item[1]
            if item[1].next:
                heapq.heappush(h,(item[1].next.val,item[1].next))
            root = root.next
        return head.next
            
    #    while heap:

        
            