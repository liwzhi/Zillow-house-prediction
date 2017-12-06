package leetcodeStudy;

public class removeLinkedListElement {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int count = 0;
        ListNode lenNode = head;
        while(lenNode!=null){
            count+=1;
            lenNode = lenNode.next;
        }
        if (count==1){
            return null;
        }
        if(count==0){
            return null;
        }
            
        
        int newIndex = count-n;
        count = 1;
        lenNode = head;
        if(count-1==newIndex){
            return head.next;
        }
        
        while(lenNode!=null && count!=newIndex){
            count+=1;
            lenNode = lenNode.next;
        }
        
        
        
        lenNode.next = lenNode.next.next;
        return head;
        
    }

}
