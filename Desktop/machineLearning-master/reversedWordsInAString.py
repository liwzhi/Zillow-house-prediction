class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return ' '.join(reversed(s.strip().split()))
        
obj = Solution()
print obj.reverseWords('a')