class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        string = '{0:32b}'.format(n)
        count = 0
        for item in string:
            if item == '1':
                count+=1
        return count