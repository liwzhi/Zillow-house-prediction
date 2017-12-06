# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        if not head:
            return 
        end = head
        while(end.next):
            end = end.next
        start = head
        endLoop = end
        pre = ListNode(0)
        pre.next = start
        result = pre
        while(start!=endLoop):
            if start.val>=x:
                nextNode = start.next
                endLoop.next = start
                pre.next = nextNode
                start = nextNode
                end = end.next
            else:
                pre = start
                start = start.next
        return result.next
                    
                
            
        
        