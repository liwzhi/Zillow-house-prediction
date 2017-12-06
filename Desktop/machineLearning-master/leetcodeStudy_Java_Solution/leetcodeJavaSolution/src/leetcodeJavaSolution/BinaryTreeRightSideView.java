package leetcodeJavaSolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class BinaryTreeRightSideView {
	public List<Integer> rightView(TreeNode root){
		List<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stk = new Stack<TreeNode>();
		stk.add(root);
		while(!stk.isEmpty()){
			Stack<TreeNode> nextLevel = new Stack<TreeNode>();
			for(TreeNode item:stk){
				if(item.left!=null){
					nextLevel.push(item.left);
				}
				if(item.right!=null){
					nextLevel.push(item.right);
				}
			}
			result.add(stk.get(stk.size()-1).val);
			stk = nextLevel;
			
		}
		
		
		return result;
	}

}
