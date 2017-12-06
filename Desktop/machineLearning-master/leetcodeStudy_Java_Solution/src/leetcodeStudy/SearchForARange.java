package leetcodeStudy;

public class SearchForARange {
	public int[] searchRange(int[] nums,int target){
		int[] result = {-1,-1};
		int high = nums.length-1;
		int low = 0;
		while(low<=high){
			int mid = low + (high-low)/2;
			if (nums[mid]==target){
				int left = mid;
				int right = mid;
				while(left>=0 &&nums[left]==target){
					left-=1;
				}
				result[0]= left+1;
				while(right<=high && nums[right]==target){
					right -=1;
				}
				result[1] = right-1;
				return result;
				
			}
			else if(nums[mid]>target){
				high = mid-1;
			}
			else{
				low = mid+1;
			}
		}
		
		return result;
	}
	
	

}
