# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:54:27 2015

@author: weizhi
"""
"""
Time exceed, no idea yet...

Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.

Note:
Have you consider that the string might be empty? This is a good question to ask during an interview.

For the purpose of this problem, we define empty string as valid palindrome.


"""



import string
class Solution:
    flag = 0
    def isPalindrome(self,s):
        itemList = list(string.ascii_lowercase)
        s = s.lower()
        print s
        element = ''
        for item in s:
            if item in itemList:
                element = element + item
        #print element
        element = list(element)
        print element
        while len(element)>1:
            start = element.pop(0)
            end = element.pop(-1)
            print element
            if start != end:
                return False
       # self.palindrome(element,self.flag)
        return True

    
    
    
    
    def palindrome(self,s,flag):
        if len(s) ==1 or len(s)==0:
            self.flag = 1
            return self.flag
        #item = [i for i in s]
       # length = len(s)
        if s[0] == s[-1]:
            self.palindrome(s[1:-1],self.flag)
        return self.flag 
       # return False

a = Solution()
print a.isPalindrome("A man, a plan, a canal: Panama")
print a.isPalindrome("race a car")

a = "I'm 19 years old"
import re
word1 = " ".join(re.findall("[a-zA-Z]+",a))
print word1


"correct solution, we did not need to store the information"
"""str.isalnum()
Return true if all characters in the string are alphanumeric and there is at least one character, false otherwise.
From: https://github.com/liwzhi/LeetCode-2/blob/master/Python/valid-palindrome.py

"""

class Solution1:
    # @param s, a string
    # @return a boolean
    def isPalindrome(self, s):
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i, j = i + 1, j - 1
        return True

