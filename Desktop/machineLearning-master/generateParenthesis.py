class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []

        self.recur(result,"",n,n)
        return result
    
    def recur(self,result,stk,left,right):
        if left ==0 and right ==0:
            result.append(stk)
        if left>0:
            self.recur(result,stk+"(",left-1,right)
        if left<right:
            self.recur(result,stk+")",left,right-1)

obj = Solution()
print obj.generateParenthesis(3)
        