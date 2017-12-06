# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 00:18:18 2015

@author: weizhi
"""

class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: Set[str]
        :rtype: int
        """
        az = [chr(i) for i in xrange(ord('a'), ord('z')+1)]
       # if beginWord ==endWord:
       #     return 2
        stk = [beginWord]
        count = 2
        path = {beginWord:None}
        #path = {}
        while endWord not in path:
            nextLevel = []
            count +=1
            values = set()

         #   print nextLevel
        #    print stk
            for item in stk:
                for index in range(len(item)):
                    char = item[index]
                    currItem = item
                    # print char
                    while currItem:
                        values.add(currItem)
                        currItem = path[currItem]
                    print values
                    for newChar in az:                    #      print newChar
                        if newChar!=char:
                            word = list(item)
                            word[index] = newChar
                            newItem = ''.join(word)

                            if newItem in wordList:
                            #    print path[item].values()                                    
                                if newItem not in values:
                                    nextLevel.append(newItem)
                                    path[newItem] = currItem
            stk = nextLevel
        return count
                
obj = Solution()

beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
print obj.ladderLength (beginWord,endWord,wordList)   