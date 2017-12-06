class Solution(object):
    value = 0
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num==None:
            return 
        A = str(num)
        result = 0
        for item in A:
            result += int(item)
       # print result
        if result>=10:
            self.addDigits(result)
        if result<10:
            self.value = result
        return self.value
        
    def addDigits2(self,num):
        if num ==None:
            return
       # result = []
        return self.recur(num,-1)
    
    def recur(self,num,result):
        #print result
        if -1<result<10:
            return result
        else:
            result = 0
            for item in str(num):
                result +=int(item)
            result = self.recur(result,result)
        return result
        
        
        
        
obj = Solution()
print obj.addDigits2(1111)