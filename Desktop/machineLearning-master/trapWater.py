class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0
        res = 0
        rightMax = [0]*len(height)
        currMax = 0
        for i in range(len(height)-1,-1,-1):
            item = height[i]
            currMax = max(item,currMax)
            rightMax[i] = currMax
          #  print rightMax
      #  rightMax = 0
        leftMax = 0
        for i in range(len(height)):
            currItem = height[i]
            leftMax = max(currItem,leftMax)
            res =res + min(leftMax,rightMax[i])-currItem
        return res
obj = Solution()
print obj.trap([0,1,0,2,1,0,1,3,2,1,2,1])
            