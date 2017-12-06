class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        Dict = {1:"",2:'abc',3:'def',4:'ghi',5:'jkl',6:'mno',7:'pqrs',8:'tuv',9:'wxyz'}
        result = []
        self.recur(digits,result,"",0,Dict)
        return result
    def recur(self,digits,result,stk,n,Dict):
        if len(digits) == n:
            result.append(stk)
        else:
            for choice in Dict[int(digits[n])]:
                self.recur(digits,result,stk+choice,n+1,Dict)
            
        
obj = Solution()
print obj.letterCombinations('23')

           