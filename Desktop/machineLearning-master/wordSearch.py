class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if len(word) ==0:
            return True
        row = len(board)
    
        col = len(board[0][0])
        result = []
        marked = [[False for i in range(col)] for j in range(row)]
        for i in range(row):
            for j in range(col):
                if board[i][0][j] == word[0]:
                    result = self.recur([word[0]],word,i,j,board,0,[],marked) 
                    marked[i][j] = True
                    if len(result)>0:
                        return True
        return False
        
    def recur(self,stk,word,i,j,Grid,start,result,marked):
        if len(stk) == len(word):
             result.append(stk)
             return result
        row = len(Grid)
        col = len(Grid[0][0])
        if i+1<row and word[start+1]==Grid[i+1][0][j] and not marked[i+1][j]:
            marked[i+1][j] = True
            self.recur(stk+[word[start+1]],word,i+1,j,Grid,start+1,result,marked)
            
        if i-1>-1 and word[start+1]==Grid[i-1][0][j] and not marked[i-1][j]:
            marked[i-1][j] = True
            self.recur(stk+[word[start+1]],word,i-1,j,Grid,start+1,result,marked)
        if j+1<col and word[start+1] == Grid[i][0][j+1] and not marked[i][j+1]:
            marked[i][j+1] = True
            self.recur(stk+[word[start+1]],word,i,j+1,Grid,start+1,result,marked)
        if j-1>-1 and word[start+1] == Grid[i][0][j-1] and not marked[i][j-1]:
            marked[i][j-1] = True
            self.recur(stk+[word[start+1]],word,i,j-1,Grid,start+1,result,marked)
        return result
board = [
  ["ABCE"],
  ["SFCS"],
  ["ADEE"]
]  
board ["aa"]
obj = Solution()
print obj.exist(board,'aa')         
        
        