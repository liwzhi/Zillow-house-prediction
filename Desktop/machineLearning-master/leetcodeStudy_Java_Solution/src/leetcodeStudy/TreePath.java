package leetcodeStudy;

import java.awt.List;
import java.util.ArrayList;

public class TreePath {
	public ArrayList<String> binaryTreePaths(TreeNode root){
		ArrayList<String> result = new ArrayList<String>();
		if (root==null)
			return result;
		return helper(root,result,String.valueOf(root.val));
	}// main function
	public ArrayList<String> helper(TreeNode root,ArrayList<String> result,String s){
		if(root.left==null && root.right==null){
			result.add(s);
			return result;
		}
		if(root.left!=null){
			this.helper(root.left, result, s + "->" + String.valueOf(root.left.val));
		}// left exist
		if(root.right!=null){
			this.helper(root.right, result, s + "->"+String.valueOf(root.right.val));
		}// right exist
		return result;
	}

}
