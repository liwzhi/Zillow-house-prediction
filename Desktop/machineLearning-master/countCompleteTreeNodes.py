class TreeNode():
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root ==None:
            return 0
        depthLeft = self.findDepth(root,0)
        depthRight = self.findDepth2(root,0)
        print depthLeft
        print depthRight
        count = 2**(depthLeft)-1
        if depthLeft ==depthRight:
            return count
        else:
            return count-1
        
                    
    
    def findDepth(self,root,depth):
        if root ==None:
            return depth
        else:
            return self.findDepth(root.left,depth+1)
    def findDepth2(self,root,depth):
        if root ==None:
            return depth
        else:
            return self.findDepth2(root.right,depth+1)
#%%
# BFS to solv this questions            
class Solution1(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        count = 0
        stk = [root]
        while(stk):
            nextLevel = []
            for item in stk:
                count +=1
                if item.left:
                    nextLevel.append(item.left)
                if item.right:
                    nextLevel.append(item.right)
            stk = nextLevel
        return count
                    
    
    def findDepth(self,root,depth):
        if root ==None:
            return depth
        else:
            return self.findDepth(root.left,depth+1)
        
root = TreeNode(0)
a1 = TreeNode(1)
a2 = TreeNode(2)
a3 = TreeNode(3)


b1 = TreeNode(4)
b2 = TreeNode(5)
b3 = TreeNode(6)            
            


root.left = a1
a1.left = a2
a2.right = a3

root.right = b1
b1.left = b2
b1.right = b3
    
obj = Solution1()
print obj.countNodes(root)