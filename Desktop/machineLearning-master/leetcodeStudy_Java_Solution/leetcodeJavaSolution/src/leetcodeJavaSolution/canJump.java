package leetcodeJavaSolution;

public class canJump {
	public boolean canJump(int[] nums){
		if(nums==null){
			return true;
		}
		int pre = nums[0];
		//int curr = 0;
		for(int i=1;i<nums.length;i++){
			if(pre<=0){
				return false;
			}
			int curr = nums[i];
			if(nums[i]<pre-1){
				curr = pre-1;
			}
			pre = curr;

		}
		return true;
		
		}
	}


