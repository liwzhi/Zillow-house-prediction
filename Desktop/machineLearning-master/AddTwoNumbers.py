# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:47:42 2015

@author: weizhi
"""
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head1 = ListNode(0)
        head2 = ListNode(0)
        end1 = ListNode(0)
        head1.next = l1
        head2.next = l2
        flag = 0
        preFlag = 0
        while(l1 and l2):
            flag = l1.val + l2.val + preFlag
            print flag
            l1.val = (l1.val + l2.val + preFlag)%10        
            if flag>=10:
                preFlag = 1
            else:
                preFlag = 0
            end1 = l1
            l1 = l1.next
            l2 = l2.next          
        if l1:
            while(preFlag==1 and l1):
                flag = l1.val + 1
                l1.val = (l1.val + 1) %10
                if flag>=10:
                    preFlag = 1
                else:
                    preFlag = 0
                l1 = l1.next
                end1 = l1
        if l2:
            while(preFlag==1 and l2):
                flag = l2.val + 1
                l2.val = (l2.val + 1) %10
                if flag>=10:
                    preFlag = 1
                else:
                    preFlag = 0
                end1.next = l2
                end1 = l2
                l2 = l2.next
            if preFlag !=1:
                end1.next = l2
      #  print preFlag
        if preFlag ==1:
            end1.next = ListNode(1)
        return head1.next
                
                
                
            
a1 = ListNode(3) 
a2 = ListNode(7)
a3 = ListNode(3)

b1 = ListNode(9)
b2 = ListNode(2)
b3 = ListNode(4)

a1.next = a2
b1.next = b2


obj = Solution()
result = obj.addTwoNumbers(a1,b1)   

while(result):
    print result.val
    result = result.next      
            
        
            
            
            
        
