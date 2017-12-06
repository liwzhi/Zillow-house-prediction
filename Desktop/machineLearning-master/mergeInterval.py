# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:49:01 2015

@author: weizhi
"""
[[1,4],[0,2],[3,5]]
# Definition for an interval.
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if len(intervals)==0:
            return 
        result = []
        indexSort = [intervals[i].start for i in range(len(intervals))]
        s = sorted(range(len(indexSort)),key=lambda k:indexSort[k])
        intervalsNew = []
        for index in s:
            intervalsNew.append(intervals[index])
           # print intervals[index].start
          #  print intervals[index].end
            
        #pre = intervalsNew.pop(0
        result = [intervalsNew[0]]
        for i in range(1,len(intervals)):
            prev = result[-1]
            print result[-1].start
            print result[-1].end
            current = intervalsNew[i]
           # print prev.end
            if current.start<=prev.end:
                prev.end = max(prev.end,current.end)
                result[-1].end = prev.end
            else:
                result.append(current)
        
        

        return result
a = Interval(1,4)
#print a.end
b = Interval(0,2)  # 0,4
c = Interval(3,5)  #0 5
d = Interval(6,9)
data = [a,b,c]
Obj = Solution()
result =  Obj.merge([a,b,c,d])
#print result[0].end