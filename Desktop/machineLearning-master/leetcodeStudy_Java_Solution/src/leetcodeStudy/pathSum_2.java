package leetcodeStudy;

import java.util.ArrayList;
import java.util.List;

public class pathSum_2 {
	public List<List<Integer>> pathSum(TreeNode root,int sum){
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> stk = new ArrayList<Integer>();
		
		
		
		
	}
	public List<List<Integer>> helper(TreeNode root,List<List<Integer>> result,List<Integer> stk,int sum){
		if(root.left==null && root.right==null && sumList(stk)== sum){
			result.add(new ArrayList<Integer>(stk));
			return result;
		}
		if (root.left==null)
			stk.add(root.val);
			helper(root.left,result,stk,sum);
		
		
		
		
		return result;
	}
	
	public Integer sumList(List<Integer> list) {
	     Integer sum= 0; 
	     for (Integer i:list)
	         sum = sum + i;
	     return sum;
	}

}
