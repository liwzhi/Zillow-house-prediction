class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n<=0:
            return 0
        result = []
        pre = str(n)[0]
        count = 1
        i = 1
        while(i<len(str(n))):
            item = str(n)[i]
            i+=1
            while(item==pre and i<len(str(n))-1):
                print i
                count+=1
                pre = item
                item = str(n)[i]
                i+=1
              #  pre = item            
            result.append(str(count)+str(item))
            
        
            count = 1
        return int(''.join(result))
                
            
obj = Solution()
print obj.countAndSay(111)