# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root ==None:
            return 
        path1 = reduce(lambda x,y:x+y,self.recur(root,[],[],p.val))
        path2 = reduce(lambda x,y:x+y,self.recur(root,[],[],q.val))
        pre = root
        while (path1 and path2):
            item1 = path1.pop(0)
            item2 = path2.pop(0)
            if item1 ==item2:
                pre = item1
            else:
                break 
        return pre
        
            
    def recur(self,root,result,stk,q):
       # print root.val
        if root.val == q:
            result.append(stk + [root])
            return result
        if root.left:
            self.recur(root.left,result,stk + [root],q)
        if root.right:
            self.recur(root.right,result,stk+[root],q)
        return result
    def printVal(self,lst):
        for item in lst:
            print item.val
    def lowestCommonAncestor2(self, root, p, q):
        if root ==None or q==None or p ==None:
            return None
        if max(p.val,q.val)<root.val:
            self.lowestCommonAncestor2(root.left,p,q)
        elif min(p.val,q.val)>root.val:
            self.lowestCommonAncestor2(root.right,p,q)
        else:
            return root
root = TreeNode(2)
root2 = TreeNode(1)
root3 = TreeNode(3)
root4 = TreeNode(4)

root.right = root2
root2.left = root3
root2.right = root4
obj = Solution()
result = obj.lowestCommonAncestor(root,root3,root4)
print 'result'
print result.val