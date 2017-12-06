# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:27:30 2015

@author: weizhi
"""

class Solution(object):
    Max = -2**(31)
    def maxPathSum(self,root):
        self.helper(root)
        return self.Max
        
    def helper(self,root):
        if root ==None:
            return 0
        leftMax = max(0,self.helper(root.left))
        rightMax = max(0,self.helper(root.right))
        self.Max = max(self.Max,root.val + leftMax+rightMax)
        
        return root.val + max(leftMax,rightMax)

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:46:23 2015

@author: weizhi
"""

class Solution(object):
    def maxPathSum(self,root):
        if root ==None:
            return 
        stk = [root]
        result = []
        while(stk):
            nextLevel = []
            for item in stk:
                if item.left !=None:
                    nextLevel.append(item.left)
                if item.right !=None:
                    nextLevel.append(item.right)
                result.append(self.getPath(item))
            stk = nextLevel
        return max(result)
                


    def getPath(self,root):
        result = []
        if root ==None:
            return result
        result = self.helper(root,result,[])
        maxSum = []
        for path in result:
            if path !=None:
                maxSum.append(self.arrayMax(path))
        if len(maxSum) == 0:
            return 
        else:
            return max(maxSum)
    def helper(self,root,result,stk):
        if root ==None:
            return result
        if root.left ==None and root.right ==None:
            result.append(stk.append(root.val))
           # return result
        stk.append(root.val)
        self.helper(root.left,result,stk)
        self.helper(root.right,result,stk)
        stk.pop()
        return result
   
        
    def arrayMax(self, nums):
        if len(nums) ==0:
            return 0
        length = len(nums)
    #    maxSum = 0
        result = [0]*length
        result[0] = nums[0]
        for i in range(1,length):
            curr = nums[i]
            if curr + result[i-1]<0:
                result[i] = curr
            result[i] = max(curr+result[i-1])
        return max(result)
            