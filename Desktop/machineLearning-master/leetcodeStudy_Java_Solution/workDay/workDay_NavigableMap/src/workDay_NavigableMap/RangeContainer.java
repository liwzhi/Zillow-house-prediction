package workDay_NavigableMap;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class RangeContainer {
	Map<Integer, ListNode> container;
	Long[] keys;
	short bar = (short)0;
	RangeContainerFactory objCon;
	RangeContainer(Map<Integer, ListNode> containerNew, Long[] Keys){
		container = containerNew;
		keys = Keys;
	}
	
	public PriorityQueue<Short> findIdsInRange(long fromValue,long toValue,boolean fromInclusive,boolean toInclusive){
		 PriorityQueue<Short> heapq = new PriorityQueue<Short>();
		 
		 
		 int leftIndex = binarySearch(keys,fromValue);  // get the low (left) bucket number
		 int rightIndex = binarySearch(keys,toValue);  // get the high(right) bucket number
		 
		 
		 
		 if(leftIndex>rightIndex){
			 System.out.println("Warning: fromValue should be less than toValue");
		 }
		 
		 
		 for(int i=leftIndex;i<=rightIndex;i++){

			 ListNode head = container.get(i);
			 while(head!=null){
				 long value = head.val;
				 short id = head.id;
				 head = head.next;
				 
				 if(value==fromValue) {
					 if(fromInclusive){
					 heapq.add(id);}
 
				 }
				 else if(value ==toValue){
					 if(toInclusive){
					 heapq.add(id);}
				 }
				 
				 else if(value>fromValue && value<toValue){
				//	 System.out.println(value);
					 heapq.add(id);
				 }
				 else{
					 continue; 
				 }		 
				 
			 }
		 }
		 
		 
		 return heapq;
	 }
	
	
	
	
	// bineary seach to get the container key (from 0, 1, 3, 4.... count)
	public int binarySearch(Long[] keys,Long target){
		if(keys ==null || keys.length ==0){
			return -1;
		}
		int l = 0;
		int r = keys.length-1;
		while(l<r){
			int mid = l + (r-l)/2;
			if(keys[mid] == target)
				return mid;
			if(keys[mid]<target)
				l = mid+1;
			else
				r = mid-1;
		}
		return l;
		
	}
//
}
