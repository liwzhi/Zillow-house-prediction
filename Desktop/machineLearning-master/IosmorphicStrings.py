class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        len1 = len(s)
        len2 = len(t)
        if len1 !=len2:
            return False
        Dict1 = self.unique(s)
        Dict2 = self.unique(t)
        if len(Dict1)!=len(Dict2):
            return False
        else:
           #  Dict3 = Dict1.sort()
          #   print Dict3
            # print Dict1.sort()
         #   print Dict2.sort()
             return sorted(Dict1)==sorted(Dict2)
        
        #    Dict2.sort()
            
      #  return True
    def unique(self,s):
        Dict = {}
        for index in range(len(s)):
            curr = s[index]
            if curr in Dict:
                Dict[curr].append(index)
            else:
                Dict[curr] = [index]
        return Dict.values()
obj = Solution()
print obj.isIsomorphic('paper','ttile')
        