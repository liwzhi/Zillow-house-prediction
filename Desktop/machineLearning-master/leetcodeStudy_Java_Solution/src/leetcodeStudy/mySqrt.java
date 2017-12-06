package leetcodeStudy;

public class mySqrt {
	public int mySqrt(int x){
		if(x<=0)
			return 0;
		int l = 1;
		int r = x;
		while(l<r){
			int mid = r + (l-r)/2;
			if (mid*mid==x)
				return mid;
			else if (mid*mid<x)
				l = mid+1;
			else
				r = mid-1;
				
		}
		return l;
	}

}


