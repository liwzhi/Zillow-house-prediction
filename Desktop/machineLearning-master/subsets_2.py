class Solution(object):
    def subsets(self,S):
        S.sort()
        solution = []
        self.Help(S,0,[],solution)
        return solution
        
    def Help(self,S,index,tempSolution,solution):
        if index ==len(S):
           # print tempSolution
            solution.append(list(tempSolution))
            print list(tempSolution)
        else:
            self.Help(S,index+1,tempSolution,solution)
            tempSolution.append(S[index])
            self.Help(S,index+1,tempSolution,solution)
            tempSolution.pop()
    
    def subsetsWithDup(self, S):
        result = []
        self.subsetsWithDupRecu(result, [], sorted(S))
        return result
    
    def subsetsWithDupRecu(self, result, cur, S):
        if len(S) == 0  and cur not in result:
            result.append(cur)
        elif S:
            self.subsetsWithDupRecu(result, cur, S[1:])
            self.subsetsWithDupRecu(result, cur + [S[0]], S[1:])
obj = Solution()
result = obj.subsets([1,2,3])
print result

result2 = obj.subsetsWithDup([1,2,3])
print result2