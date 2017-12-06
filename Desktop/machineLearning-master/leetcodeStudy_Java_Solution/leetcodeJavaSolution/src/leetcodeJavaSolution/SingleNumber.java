package leetcodeJavaSolution;

import java.util.HashMap;

public class SingleNumber {
	public int singleNumber(int[] nums){
		HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
		int res = Integer.MAX_VALUE;
		for(Integer item:nums){
			if (map.containsKey(item)){
				map.remove(item);
			}else{
				map.put(item,1);
			}
		}
		
		for(Integer key:map.keySet()){
			 res =  map.get(key);
		}
		
		return res;
		
	}

}
