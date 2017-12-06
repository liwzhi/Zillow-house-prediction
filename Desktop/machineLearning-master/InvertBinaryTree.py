class Solution():
    def invertTree(self,root):
        if root ==None:
            return 
        head = root
        newNode = root.left
        root.left = root.right
        root.right = newNode
        self.invertTree(root.left) 
        self.invertTree(root.right)
        return head
        