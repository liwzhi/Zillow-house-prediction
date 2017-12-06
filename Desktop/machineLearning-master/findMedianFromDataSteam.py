# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:25:16 2015

@author: weizhi
"""


import heapq
class MedianFinder:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.h = []
        

    def addNum(self, num):
        """
        Adds a num into the data structure.
        :type num: int
        :rtype: void
        """
        heapq.heappush(self.h,num)
        

    def findMedian(self):
        """
        Returns the median of current data stream
        :rtype: float
        """
        if not self.h:
            return 
        if len(self.h)%2 ==1:
            return heapq.nlargest(len(self.h)//2+1,self.h)[-1]
        else:
            return sum(heapq.nlargest(len(self.h)//2,self.h)[-2:])/2.
        

# Your MedianFinder object will be instantiated and called as such:
mf = MedianFinder()
mf.addNum(1)
print mf.findMedian()
mf.addNum(1)
print mf.findMedian()
mf.addNum(2)
print mf.findMedian()
