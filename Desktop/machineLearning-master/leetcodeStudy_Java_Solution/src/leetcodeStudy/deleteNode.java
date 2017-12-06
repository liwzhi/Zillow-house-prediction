package leetcodeStudy;
public class ListNode{
	int val;
	ListNode next;
	ListNode(int x){
		val = x;
	}
}


public class deleteNode {
	public void deleteNode(ListNode node){
		  ListNode nextNode = node.next;
		  node.val = nextNode.val;
		  node.next = nextNode.next;
		  
		  
	}

}
