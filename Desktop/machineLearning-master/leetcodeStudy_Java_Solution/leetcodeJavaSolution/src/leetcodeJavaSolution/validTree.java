package leetcodeJavaSolution;

public class validTree {
	public boolean isValidTree(TreeNode root){
		int maxValue = Integer.MAX_VALUE;
		int minValue = Integer.MIN_VALUE;
		return helper(root,maxValue,minValue);
	}
	boolean helper(TreeNode root,int maxValue,int minValue){
		if(root==null){
			return true;
		}if(root.val>=maxValue || root.val<=minValue){
			return false;
		}
		return helper(root.left,root.val,minValue) && helper(root.right,maxValue,root.val);
		
	}

}
