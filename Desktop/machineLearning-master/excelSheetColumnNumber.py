class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n<1:
            return 
        index = str(n/26)
        left = index + str(n%26)
        if n%26 ==0:
            
        
        
        result = []
        for item in left:
            if item !='0':
                result = result + [chr(64+int(item))]
        return "".join(result)
#    def convertToTitle2(self, n):
#        """
#        :type n: int
#        :rtype: str
#        """
#        if not n:
#            return 
#        result = 0
#        for item in 
obj = Solution()
print obj.convertToTitle(26)

        
            
        