package leetcodeJavaSolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class perumutation {
	public List<List<Integer>> permute(int[] nums){
		List<List<Integer>> result = new ArrayList<List<Integer>>();

		
		List<Integer> stk = new ArrayList<Integer>();
		boolean[] flag = new boolean[nums.length];
		for(int i=0;i<nums.length;i++){
			flag[i]= false;
		}
		 helper(nums,result,stk,flag);
		 return result;
	}
	public void helper(int[] nums,List<List<Integer>> result,List<Integer> stk,boolean[] flag){

		if(stk.size()==nums.length){
			System.out.println(stk);

			result.add(new ArrayList<Integer>(stk));
		//	System.out.println(result);

			//return result;
		}
		for(int i=0;i<nums.length;i++){
			
			if(!flag[i]==true){
				stk.add(nums[i]);
				flag[i] =  true;
				helper(nums,result,stk,flag);
				flag[i] = false;
				stk.remove(stk.size()-1);
			}
			
		}

		//return result;
		
	}
	

}
