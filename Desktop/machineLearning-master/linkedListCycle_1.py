class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return False
        slow = head
        fast = head
        while(fast and slow and fast.next):
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False