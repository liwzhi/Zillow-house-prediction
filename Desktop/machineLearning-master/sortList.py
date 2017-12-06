# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:41:02 2015

@author: weizhi
"""
class listNode():
    def __init__(self,x):
        self.val = x
        self.next = None
class Solution(object):
    def sortList(self,head):
        if head ==None or head.next ==None:
            return head
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        head1 = head
        head2 = slow.next
        slow.next = None
        head1 = self.sortList(head1)
        head2 = self.sortList(head2)
        head = self.merge(head1,head2)
        return head
    
    
    def merge(self,headFirst,headSecond):
        if headFirst==None:
            return headSecond
        if headSecond ==None:
            return headFirst
        dummy = ListNode(0)
        curr = dummy
       # head = headFirst
       # curr = head
        while headFirst and headSecond:
            if headFirst<=headSecond:
                curr.next = headFirst
                headFirst = headFirst.next                            
            else:
                curr.next = headSecond
                headSecond = headSecond.next
            curr = curr.next        
        if headSecond==None:
            curr.next = headFirst
        if headFirst==None:
            curr.next = headSecond
          #  curr = curr.next
           # headSecond = headSecond.next
        return dummy.next 
            
a1 = listNode(2)
a2 = listNode(1)
a3 = listNode(0)
a1.next = a2   
a2.next = a3  
Obj = Solution()
a1 = Obj.sortList(a1)
while(a1):
    print a1.val
    a1 = a1.next


      
