package leetcodeStudy;

import java.awt.List;
import java.util.ArrayList;

public class minimumTotal {
    public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
    	if(triangle==null || triangle.size()==0)
    		return 0;
    	if(triangle.size()==1)
    		return triangle.get(0).get(0);
    	int[] sums = new int[triangle.size()];
    	sums[0] = triangle.get(0).get(0);
    	for(int i=1;i<triangle.size();i++){
    		sums[i] = sums[i-1] + triangle.get(i).get(i);
    		for(int j=i-1;j>=1;j--)
    		{
    			sums[j] = (sums[j]<sums[j])
    		}
    	}// main for loop
    	
    	
    }

}
