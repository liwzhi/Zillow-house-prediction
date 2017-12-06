package workDay_NavigableMap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

public class RangeContainerFactory {
	
	public long diff; 
	public long minValue = Long.MIN_VALUE;
	public long maxValue = Long.MAX_VALUE;
	public HashMap<Integer,Long> keyToIncomeMap;
	public Long[] Keys;
	RangeContainerFactory(long diffValue){
		diff = diffValue;
	}
	
	public Map<Integer, ListNode> createContainer(long[] data){
		
		
		Map<Integer, ListNode> container = new HashMap<Integer,ListNode>();
		
		
		
		long maxValue = max(data);
		long minValue = min(data);
		
		int count = (int) ((maxValue-minValue)/diff);

		Long[] keys = new Long[count];
		
		
		for (int i = 0;i<count;i++){
			keys[i] = minValue + (long)(diff*i);

		}
		// save the key ---> income information
		keyToIncomeMap = keyToIncome(keys);
		Keys = keys;   // save the key

		// put the value into different bucket 
		
		for(int i=0;i<data.length;i++){
			short id = (short)(i);
			long item = data[i];
			int key;
			if(item>maxValue){
				key = keys.length-1;   // new item value is bigger than the exist value
			}
			else if(item<minValue){
				key = 0;     // new item value is less than the exist value
			}
			else{
			key = binarySearch(keys,item); // get the bucket key
			}
			
			ListNode newNode = new ListNode(id,item);
			
			
			if(container.get(key)==null){
				container.put(key, newNode);
				
			}else{
				ListNode existNode = container.get(key);
	
				ListNode updateNode = addNode(existNode,newNode);
				container.put(key, updateNode); // get the update head of list in special key 
				
			}
			
			
			
		}
		
		
		return container;
		
		
	}
	
	// add new value into the container
	public ListNode addNode(ListNode head, ListNode newNode){
		if(head ==null){
			return newNode;
		}
		
		ListNode dummy = new ListNode((short)(-1),Long.MIN_VALUE);
		dummy.next = head; // keep the previous node information 
		
		ListNode dummyHead = head;
		int count = 0;
		while(dummyHead!=null && newNode.val>dummyHead.val){
			dummyHead = dummyHead.next;
			dummy = dummy.next;
			count +=1;
		}
		if(count==0){
			newNode.next= head;
			
			return newNode;
		}
		if(dummyHead==null){
			dummy.next = newNode;
			
		}
		if(count!=0 && dummyHead!=null){
			dummy.next = newNode;
			newNode.next = dummyHead;
		}

		
		return head;
	}
	// delete the exist value (income) in the container
	public ListNode deleteNode(ListNode head, ListNode delNode){
		
		if(delNode==null){
			return head;
		}

		
		ListNode dummy = new ListNode((short)(-1),Long.MIN_VALUE);
		dummy.next = head; // keep the previous node information 
		
		ListNode dummyHead = head;
		int count = 0;
		while(delNode!=dummyHead){
			dummyHead = dummyHead.next;
			dummy = dummy.next;
			count +=1;
		}
		
		
		if(count==0){
			
			return delNode.next;
		}
		else{
			ListNode nextNode = delNode.next;
			dummy.next = nextNode;
		}
		return head;
	}

	
	// get the key to map the income: 0:minvalue,  1: minVaue + diff, 2: minValue+2*diff + ...
	
	public HashMap<Integer,Long> keyToIncome(Long[] keys){
		HashMap<Integer,Long> res = new HashMap<Integer,Long>();
		for(int i=0;i<keys.length;i++){
			res.put(i, keys[i]);
		}
		
		return res;
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
	
	
	// get the maxvalue in data set
	public long max(long[] data){
		long maxValue = Long.MIN_VALUE;
		for (Long item:data){
			maxValue = Math.max(maxValue,item);
		}
		return maxValue;
	}
	
	// get the minvalue in data set
	public long min(long[] data){
		long minValue = Long.MAX_VALUE;
		for (Long item:data){
			minValue = Math.min(minValue,item);
		}
		return minValue;
	}

}
