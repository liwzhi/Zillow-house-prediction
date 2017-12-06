package leetcodeJavaSolution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

public class postOrderTraversal {
	public List<Integer> postOrder(TreeNode root){
		
		List<Integer> result = new ArrayList<Integer>();
		if(root==null){
			return result;
		}
		HashMap<TreeNode,Boolean> marked = new HashMap<TreeNode,Boolean>();
		Stack<TreeNode> stk = new Stack<TreeNode>();
		stk.push(root);
		while(!stk.isEmpty()){
			TreeNode item = stk.pop();
			if(!marked.containsKey(item)){
			marked.put(item, true);
			stk.push(item);
			if(item.right!=null){
				stk.push(item.right);
			}
			if(item.left!=null){
				stk.push(item.left);
			}}
			else{
				result.add(item.val);
			}
			
		}
		
		return result;
	}

}
