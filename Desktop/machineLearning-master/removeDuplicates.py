# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:24:24 2015

@author: weizhi
"""

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head ==None:
            return 
        pre = ListNode(0)
        pre.next = head
        result = pre
        curr = head
        while(curr.next):
            nextNode = curr.next
            if curr.val.val == nextNode.val:
                pre.next = nextNode
                curr = nextNode
            else:
                pre = curr
                curr = nextNode
        return result.next
                