package leetcodeStudy;

import java.util.*;
public class treeTraversal {
	List<Integer> resultRec = new ArrayList<Integer>();
	public List<Integer> treePreorder(TreeNode root){
		List<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> infor = new Stack<TreeNode>();
		while (root!=null || !infor.isEmpty()){
			if (root!=null){
				infor.push(root);
				root = root.left;}
			else{
				root = infor.pop();
				result.add(root.val);
				root = root.right;
			}
				
			
		}
	
		
		return result;
	}
	
	public List<Integer> inorderRec(TreeNode root){
		if(root !=null)
			helper(root);
		
		 return resultRec;
	}

	private void helper(TreeNode root) {
		// TODO Auto-generated method stub
		if(root.left!=null){
			helper(root.left);
		}
		resultRec.add(root.val);
		if(root.right!=null){
			helper(root.right);
		}
		
	}


}
