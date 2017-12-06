# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    flag = 0
    def reorderList(self,head):
        if not head:
            return head
        dummy = head
        pre = head 
        curr = pre.next
        end = self.findTheEnd(pre)
        while(curr!=end):
            end = self.findTheEnd(pre)
            last = end.next
            if last ==None:
               # return dummy
               break
            else:
                pre.next = last
                last.next = curr
                pre = curr
                curr = curr.next
                end.next = None
      #  return dummy
    def findTheEnd(self,head):
      #  if not head:
     #       return head
       # if head.next ==None or head.next.next ==None:
         #   return head
        if head ==None:
            return head
        if head.next ==None:
            return head
        flag = 0
        while(head.next.next):
            head = head.next
            flag = 1
        if flag ==1:
            return head
        else:
            return head.next
        
    #    return head.next        
