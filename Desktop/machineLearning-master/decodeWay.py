class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s ==None:
            return 

        if len(s) ==0 or int(s[0])==0:
            return 0
        result = [1 for i in range(len(s))]
        result[0] = 1
        for index in range(1,len(s)):
            preItem = int(s[index-1])
            item = int(s[index])
            Sum = preItem*10 + item
            if item ==0:
                if preItem ==0:
                    return 0
                if Sum>26:
                    return 0
                if index ==len(s)-1 and Sum>26:
                    return 0
                if index==len(s)-1 and Sum<26:
                    result[index] = result[index-2]
            else:
                if 0<Sum<=26 and preItem!=0:
                    result[index] = result[index-1] + result[index-2]       
                elif Sum>26 and preItem!=0:
                    result[index] = result[index-1]
                else:
                    result[index] = result[index-2]
                
            
      #  print result
        return result[-1]
obj = Solution()
result = obj.numDecodings('16205')
print result