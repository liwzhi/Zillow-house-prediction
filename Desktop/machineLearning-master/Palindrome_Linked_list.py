# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:50:08 2015

@author: weizhi
"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    flag = 0
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head==None or head.next==None:
            return True
     #   if head.next.next==None and head.val!=head.next.val:
     #       return False
        fast = head
        slow = head
        while(fast.next and fast.next.next):
            fast = fast.next.next
            slow = slow.next
        
        head1 = slow.next
        head2 = self.reverseList(head1)
        
      #  self.print1(head2)
        slow.next = None
        slow = head
        
      #  self.print1(head2)
        while head2:
            if slow.val!=head2.val:
                self.flag = 1
                break
            head2 = head2.next
            slow = slow.next
        return self.flag ==0
            
        
    def reverseList(self,head):
        if head==None:
            return head
        pre = None
        curr = head
        while curr:
            nextNode = curr.next
            curr.next = pre
           # pre = curr.next
            pre = curr
            curr = nextNode
        return pre
    def print1(self,head):
        while head:
            print head.val
            head = head.next
            
            