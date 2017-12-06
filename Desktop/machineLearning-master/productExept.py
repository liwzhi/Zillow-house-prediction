# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:49:16 2015

@author: weizhi
"""
#%% Solution one is not allowed


class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return 
        count =0
        result = 1
        for index in range(len(nums)):
            item = nums[index]
            result = result*item
          #  result[index] = reduce(lambda: x,y:x*y,result[:index],item)
            if item==0:
                count+=1
        res =[1]*len(nums)
        if count>1:
            return [0]*len(nums)
        if count ==1:
            index = nums.index(0)
            res = [0]*len(nums)
            nums.remove(0)
         #   print nums
            res[index]= reduce(lambda x,y:x*y,nums,1)
        #    return res
        if count ==0:
            for i in range(len(nums)):
                res[i] = result/nums[i]
        return res
        
    def productExceptSelf2(self,nums):
        if not nums:
            return
        result1 = [1]*len(nums)
        result2 = [1]*len(nums)
        for index,content in enumerate(nums):
            if index ==0:
                continue 
            else:
                item= nums[:index]
                result1[index] = reduce(lambda x,y:x*y,item,1)
        print result1
                
        for index,content in enumerate(reversed(nums)):
            if index ==0:
                result2[len(nums)-1] = nums[index]*result1[len(nums)-1] 
            else:
                item= nums[-index:]
                #print item
                result2[len(nums)-index-1] = reduce(lambda x,y:x*y,item,result1[len(nums)-1-index])
        return result2
                            
   
   
        
obj = Solution()
print obj.productExceptSelf([1,2,3,4])
print obj.productExceptSelf([0,1,2,3,4])
print obj.productExceptSelf2([1,2,3,4])
            
        
        