package leetcodeStudy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class subset_1 {
	public List<List<Integer>> subsets(int[] nums){
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> stk = new ArrayList<Integer>();
		Arrays.sort(nums);
		return helper(nums,result,stk,0);
	//	return result;
		
	}
public List<List<Integer>> helper(int[] nums,List<List<Integer>> result,List<Integer> stk,int start){

		if(start ==nums.length){
		  //  List<Integer> item = new ArrayList<Integer>(stk);
		    if (!result.contains(new ArrayList<Integer>(stk)))
			result.add(new ArrayList<Integer>(stk));

			return result;
		}
		for(int i=start;i<nums.length;i++){
			helper(nums,result,stk,i+1);
			stk.add(nums[i]);
			helper(nums,result,stk,i+1);
			stk.remove(stk.size()-1);
		}
		return result;
		
	}

}
