class Solution(object):
    def reverseBits(self, n):
        """
        :type n: int
        :rtype: int
        """
      #  number = '{0:32b}'.format(n).zfill(32)
        #number = bin(n)[2:].zfill(32)[::-1]
       # print number
       # number = number[::-1]
       # print number
        return int(bin(n)[2:].zfill(32)[::-1],2)
obj = Solution()
print obj.reverseBits(43261596)