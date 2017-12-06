package leetcodeJavaSolution;

public class climbStairs {
	public int climbStairs(int n){
		int result = 0;
		int[] dp = new int[n+1];
		if(n==0){
			return result;
		}
		dp[1] = 1;
		if(n==1){
			return dp[1];
		}
		dp[2] = 2;
		if(n==2){
			return dp[2];
		}
		for(int i=3;i<=n;i++){
			dp[i] = dp[i-1] + dp[i-2];
		}
		return dp[n];
		
	}

}
