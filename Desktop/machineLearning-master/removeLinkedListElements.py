# Definition for singly-linked list.
class ListNode(object):    
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if head ==None:
            return 
        pre = ListNode(0)
        pre.next = head
        result = pre
        curr = head
        while(curr):
            nextNode = curr.next
            if curr.val == val:
                pre.next = nextNode
                curr = nextNode
            else:
                pre = curr
                curr = nextNode
        return result.next
            
                
        