# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.left =None
        self.right = None

class Solution(object):
    result =[]
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        
        if len(preorder)==0 or len(inorder)==0:
            return None
        #val = inorder.pop(0)
       # print val
        mid = preorder.index(inorder[0])
        print mid
        print preorder
        print inorder
        root = TreeNode(inorder[0])
        root.left = self.buildTree(preorder[:mid],inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:],inorder[mid+1:])
        return root
    def buildTree1(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if len(preorder)==0 and len(inorder)==0:
            return None
       # if len(preorder)==1 and len(inorder)==1:
        #    return 
        curr = preorder.pop(0)
        mid = inorder.index(curr)

        root = TreeNode(curr)
        if mid>=2:
            root.left = self.buildTree(preorder[:mid],inorder[:mid-1])
        elif mid ==1:
            root.left = TreeNode(preorder[0])
        else:
            return root
        if mid+1 < len(preorder):
            root.right = self.buildTree(preorder[mid+1:],inorder[mid:])
        else:
            root.right = TreeNode(preorder[-1])
        return root
        
        
        
        
        
    def buildTree2(self, preorder, inorder):
        lookup = {}
        for i, num in enumerate(inorder):
            lookup[num] = i
        return self.buildTreeRecu(lookup, preorder, inorder, 0, 0, len(inorder))
    
    def buildTreeRecu(self, lookup, preorder, inorder, pre_start, in_start, in_end):
        if in_start == in_end:
            return None
        node = TreeNode(preorder[pre_start])
        i = lookup[preorder[pre_start]]
        node.left = self.buildTreeRecu(lookup, preorder, inorder, pre_start + 1, in_start, i)
        node.right = self.buildTreeRecu(lookup, preorder, inorder, pre_start + 1 + i - in_start, i + 1, in_end)
        return node
obj = Solution()
#root = obj.buildTree2([1,2,3],[2,3,1])
root2 = obj.buildTree1([1],[1])
print root












