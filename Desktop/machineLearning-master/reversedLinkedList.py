# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = head
        curr = pre.next
        if pre ==None or curr.next==None:
            return head
        while curr:
            nextNode = curr.next
            curr.next = pre
            pre.next = nextNode
            pre = curr
            curr = pre.next
        return p
            
            

