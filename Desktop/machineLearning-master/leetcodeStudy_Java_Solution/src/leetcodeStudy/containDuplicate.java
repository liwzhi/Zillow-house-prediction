package leetcodeStudy;

import java.util.HashSet;

public class containDuplicate {
	public boolean containsDuplicate(int[] nums){
		if (nums==null || nums.length ==0)
			return false;
		HashSet<Integer> set = new HashSet<Integer>();
		for (int i:nums){
			if (!set.add(i)){
				return true;
			}
		}// for loop first
		return false;
	}// class level

}
