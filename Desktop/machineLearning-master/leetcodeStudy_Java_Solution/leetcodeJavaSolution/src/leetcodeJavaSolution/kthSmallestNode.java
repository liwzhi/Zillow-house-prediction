package leetcodeJavaSolution;

import java.util.Stack;

public class kthSmallestNode {
	public int kthSmallest(TreeNode root,int k){
		if(root==null)
			return Integer.MAX_VALUE;
		int count = 0;
		int result = Integer.MAX_VALUE;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		while(root!=null || !stk.isEmpty()){
			if(root!=null){
				stk.add(root);
				root = root.left;
				
			}else{
				root = stk.pop();
				count+=1;
				if (count==k){
					return root.val;
				}
				root = root.right;
			}
		}
		
		return result;
		
	}

}
