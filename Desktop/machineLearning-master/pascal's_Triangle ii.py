# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:11:42 2015

@author: weizhi
"""

class Solution(object):
    def getRow(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows==0:
            return [1]
       # preLevel = [1]
        #result.append(preLevel)
     #   if numRows ==:
      #      return preLevel
        for level in range(1,numRows+1):
            
            currLevel = [1 for i in range(level+1)]
            for index in range(1,level):
                currLevel[index] = preLevel[index-1] + preLevel[index]
            preLevel = currLevel
           # result.append(preLevel)
        return currLevel
            