# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:32:18 2015

@author: weizhi
"""

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits)==0:
            return []
        Dict = {'1':[],'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'xywz'}
        result = []
        self.helper(digits,result,'',0,Dict)
        return result
    
    def helper(self,Digits,result,stk,start,Dict):
        if  len(stk) == len(Digits):
            result.append(stk[:])
        for i in range(start,len(Digits)):
            digit = Digits[i]
            for item in Dict[digit]:
                self.helper(Digits,result,stk+item,i+1,Dict)
                
        