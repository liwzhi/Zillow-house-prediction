# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:01:43 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:59:53 2015

@author: weizhi
"""

import heapq
 

a = [3,4,5,6,709,88.90]
b = [1,30,49,19,39,49]
h = []
for i in range(len(a)):
    heapq.heappush(h,a[i])
result = []
while h:
    result.append(heapq.heappop(h))
#print result



# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        h = []
        
        head = ListNode(0)
        root = head
        for node in lists:
            if node:
                heapq.heappush(h,(node.val,node))
        while h:
            item = heapq.heappop(h)
            root.next= item[1]
            if item[1].next:
                heapq.heappush(h,(item[1].next.val,item[1].next))
            root = root.next
        return head.next
            
    #    while heap:

import heapq
class MedianFinder:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.small = [-2**(31)]
        self.large = [2**(32)]
        self.mid = 0
        self.mark1 = 0
        self.mark2 = 0
        

    def addNum(self, num):
        """
        Adds a num into the data structure.
        :type num: int
        :rtype: void
        """
        
        smallValue = heapq.nlargest(1,self.small)
        largestValue = heapq.nsmallest(1,self.large)
        
        if num>=largestValue:
            heapq.heappush(self.large,num)
            self.mark1 +=1
        else:
            heapq.heappush(self.small,num)
            self.mark2+=1
            
        if self.mark1==2:
            temp = heapq.heappop(self.large)
            heapq.heappush(self.small,temp)
            self.mark1=0
        if self.mark2==2:
            temp = heapq.nlargest(1,self.small)[0]

            self.small.remove(temp)
            heapq.heapify(self.small)
            heapq.heappush(self.large,temp)
            self.mark2=0
            

    def findMedian(self):
        """
        Returns the median of current data stream
        :rtype: float
        """
        if len(self.small)==1 and len(self.small)==1:
            return 
        maxValue = heapq.nsmallest(1,self.large)
        minValue = heapq.nlargest(1,self.small)
        if len(self.small)>len(self.large):
            return minValue
        elif len(self.small)==len(self.large):
            return sum(maxValue+minValue)/2.
        else:
            return maxValue
        

# Your MedianFinder object will be instantiated and called as such:
mf = MedianFinder()
mf.addNum(1)
print mf.findMedian()        
mf.addNum(2)

print mf.findMedian()        

mf.addNum(1)
print mf.findMedian()        

mf.addNum(1)
print mf.findMedian()        
                  