package leetcodeStudy;

import java.util.Arrays;
import java.util.Set;

public class wordBreak {
	public boolean wordBreak(String s,Set<String> wordDict){
		if (s==null)
			return true;
		int strLen = s.length();
		boolean[] array = new boolean[strLen];
		Arrays.fill(array, Boolean.FALSE);
		for (int i=0;i<strLen;i++){
			String item = s.substring(0, i);
			if (wordDict.contains(item)){
				array[i] = true;
				int pre = i;
				int j = i;
				while(j<strLen){
					String item2 = s.substring(pre, j);
					if (wordDict.contains(item2) && array[pre]){
						array[j]= true;
						pre = j;
					}
				}// from j:end
			}// item in the set
			
		}// main loop
			
		//System.out.println(array[strLen-1]);;
		return array[strLen-1];
	}
	

}
