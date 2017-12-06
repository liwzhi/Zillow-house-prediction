class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return 

        slow = head
        fast = head
        marked = None
        while(fast and slow and fast.next):
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                fast = head
                while(fast!=slow):
                    fast = fast.next
                    slow = slow.next
                return slow
        return None
            