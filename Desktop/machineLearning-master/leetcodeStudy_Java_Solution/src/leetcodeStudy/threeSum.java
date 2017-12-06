package leetcodeStudy;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class threeSum {
	public List<List<Integer>> threeSum(int[] nums){
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		int begin = 0;
		int end = nums.length-1;
		int mid = 0;
		while(begin<end){
			List<Integer> curr = new ArrayList<Integer>();
			mid = begin+1;
			while(mid<end){
			int value = nums[begin] + nums[end] + nums[mid];
			if (value==0){
				curr.add(begin);
				curr.add(end);
				curr.add(mid);
				result.add(curr);
			}			
			if(value>0){
				end-=1;
			}
			if(value<0){
				mid+=1;
			}

			
			}
			begin+=1;
			
		}
		
		
		return result;
		
	}


}
