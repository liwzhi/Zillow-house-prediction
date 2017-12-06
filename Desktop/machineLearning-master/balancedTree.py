class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root ==None:
            return True
        result = self.helper(root)
        return result>0
        
    def helper(self,root):
        if root ==None:
            return 0
        leftDepth = self.helper(root.left)
        rightDepth = self.helper(root.right)
        if abs(leftDepth-rightDepth)>1 or leftDepth<0 or rightDepth<0:
            return -1
        return max(leftDepth,rightDepth) +1