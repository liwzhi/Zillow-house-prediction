package leetcodeJavaSolution;

public class containerWithMostWater {
	public int maxArea(int[] height){
		int result = 0;
		int left = 0;
		int right = height.length-1;
	
			while(right>left){
			int diff = right-left;
			int currHeight = Math.min(height[right], height[left]);
			int maxArea = currHeight*diff;
			result = Math.max(result, maxArea);
			if(height[left]>height[right]){
			   right-=1;
			}
			else{
			    left+=1;
			}
			}
		
		return result;
		
	}

}
