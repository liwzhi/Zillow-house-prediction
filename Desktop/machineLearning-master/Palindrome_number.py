class Solution:
    # @return a boolean
    def isPalindrome(self, x):
        if x < 0:
            return False
        copy, reverse = x, 0
        
        while copy:
           # print reverse
            reverse =reverse* 10
            reverse = reverse + copy % 10
            copy = copy / 10
            #print copy
            print reverse
        
        return x == reverse

if __name__ == "__main__":
    print Solution().isPalindrome(22322)
    #print Solution().isPalindrome(12320)
    #print Solution().isPalindrome(-12321)