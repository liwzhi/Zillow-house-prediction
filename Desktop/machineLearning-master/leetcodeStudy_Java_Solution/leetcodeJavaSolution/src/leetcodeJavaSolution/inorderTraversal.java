package leetcodeJavaSolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class inorderTraversal {
	public List<Integer> inorderTraversal(TreeNode root){
		List<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stk = new Stack<TreeNode>();
		while(root!=null || !stk.isEmpty()){
			if(root!=null){
				stk.push(root);
				root = root.left;
				
			}else{
				root = stk.pop();
				result.add(root.val);
				root = root.right;
			}
			
		}
		
		return result;
		
		
	}

}
