package leetcodeJavaSolution;

public class mainRun {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		perumutation obj1 = new perumutation();
		int[] nums = {1,2,3,4};
		System.out.println(obj1.permute(nums));
		System.out.println("hello world");
		// 
		combinationSum objSum = new combinationSum();
		int[] nums2 = {5,10,8,4,4,1,2,9};
		System.out.println(objSum.combinationSum(nums2, 10));
		
		// pasical triangle
		pasicalTriangle objePas = new pasicalTriangle();
		System.out.println(objePas.generate(10));
		// jump game 2
		JumpGameTwo obje2 = new JumpGameTwo();
		int[] nums3 = {2,3,1,1,4};
		System.out.println(obje2.jump(nums3));
		}

	}


