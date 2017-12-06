# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:05:50 2015

@author: weizhi
"""

class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        result = []
        if numRows<1:
            return result
        preLevel = [1]
        result.append(preLevel)
        if numRows ==1:
            return result
        for level in range(2,numRows+1):
            
            currLevel = [1 for i in range(level)]
            for index in range(1,level-1):
                currLevel[index] = preLevel[index-1] + preLevel[index]
            preLevel = currLevel
            result.append(preLevel)
        return result
            