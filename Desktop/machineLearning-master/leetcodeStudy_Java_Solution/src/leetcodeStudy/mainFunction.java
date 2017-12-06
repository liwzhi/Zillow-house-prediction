package leetcodeStudy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class mainFunction {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("hello world");
		containDuplicate obj = new containDuplicate();
		int[] nums={1,2,3,4,5,5};
		System.out.println(obj.containsDuplicate(nums));
		
		TreeNode n1 = new TreeNode(0);
		TreeNode n2 = new TreeNode(1);
		TreeNode n3 = new TreeNode(3);
		n1.left = n2;
		n1.right = n3;
		
		// has path sum
		pathSum objPath = new pathSum();
		System.out.println("has Path sum outputs");
		System.out.println(objPath.hasPathSum(n1,4));
		System.out.println(objPath.hasPathSum(n1,3));
		// has fibonacci series
		climbStairs objClm = new climbStairs();
		//System.out.format("the result is %d",objClm.climbStairs(10));
		// lists
		ArrayList<Integer> alist = new ArrayList<Integer>();
		alist.add(1);
		alist.add(3);
		System.out.format("the string is %d\n ", alist.get(0));
		HashSet<String> aSet = new HashSet<String>();
		aSet.add("bbbb");
		aSet.add("cccc");
		Iterator<String> iterator = aSet.iterator();
		// insert search
		searchInsert objInsert = new searchInsert();
		int[] numsA = {1,2,3,4,5,6,7,80,100,300};
		int target = 10;
		System.out.format("the index is %d\n", objInsert.searchInsert(numsA, target));
		
		mySqrt objSqrt = new mySqrt();
		System.out.format("the value is %d\n", objSqrt.mySqrt(101));
		
		// word break
		wordBreak objBreak = new wordBreak();
		String s = "leetcode";
		Set<String> Dict = new HashSet<String>();
		Dict.add("leet");
		Dict.add("code");
		//System.out.println("word break");
		System.out.print(objBreak.wordBreak(s, Dict));
		
		// tree path sum
		SumRootToLeafNumbers objPath2 = new SumRootToLeafNumbers();
		System.out.println("the output is\n");
		System.out.println(objPath2.sumNumbers(n1));
		System.out.println("word break");
		
	
		
		
		
		

	}

}
