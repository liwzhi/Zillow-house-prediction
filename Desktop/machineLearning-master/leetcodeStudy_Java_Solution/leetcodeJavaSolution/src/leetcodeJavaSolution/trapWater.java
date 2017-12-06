package leetcodeJavaSolution;

public class trapWater {
	public int trap(int[] height){
		int result = 0;
		int[] right = new int[height.length];
		int maxValue = Integer.MIN_VALUE;
		for(int i=height.length-1;i>-1;i--){
			maxValue=Math.max(height[i],maxValue);
			right[i]= maxValue;
			
		}
		maxValue = Integer.MIN_VALUE;
		int depth = 0;
		for(int i=0;i<height.length;i++){
			maxValue = Math.max(height[i], maxValue);
		depth = Math.min(maxValue, right[i]);
		int diff = depth-height[i];
		if(diff>0){
			result +=diff;
		}	
		}	
		return result;
	}

}
