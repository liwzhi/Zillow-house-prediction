package leetcodeStudy;

public class addTwoNumbers {
	public ListNode addTwoNumbers(ListNode l1,ListNode l2){
		ListNode result = new ListNode(0);
		ListNode dummy = result;
		int nextCount = 0;
		int count = 0;
		while(l1!=null && l2!=null ){
			
			int sumValue = l1.val+l2.val+nextCount;
			count = sumValue%10;
			if (sumValue>=10){
				nextCount = 1;
			}
			else{
				nextCount = 0;
			}
			l1.val = count;
			dummy.next = l1;
			l1 = l1.next;
			l2= l2.next;
			dummy = dummy.next;
			
		}
		int sumValue = 0;
		while(l1!=null){
			sumValue = l1.val + nextCount;
			count = sumValue%10;
			if(sumValue>=10){
				nextCount = 1;
			}
			else{
				nextCount = 0;
			}
			l1.val = count;
			dummy.next = l1;
			l1 = l1.next;
			dummy = dummy.next;
		}
		while(l2!=null){
			sumValue = l2.val + nextCount;
			count = sumValue%10;
			if(sumValue>=10){
				nextCount = 1;
			}
			else{
				nextCount = 0;
			}
			l2.val = count;
			dummy.next = l2;
			l2 = l2.next;	
			dummy = dummy.next;
			}
		if (nextCount==1){
			ListNode newNode = new ListNode(1);
			dummy.next = newNode;
		}
		
		
		return result;
		
	}

}
