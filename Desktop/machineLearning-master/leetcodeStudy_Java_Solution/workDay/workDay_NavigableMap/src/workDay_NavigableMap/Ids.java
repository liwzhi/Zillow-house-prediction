package workDay_NavigableMap;

import java.util.PriorityQueue;

public class Ids {
	
	
	PriorityQueue<Short> heapq;

	Ids(PriorityQueue<Short> h){
		heapq = h;
	}
	public short nextId(){
		
		if(!heapq.isEmpty()){
			 return heapq.poll();
		}
		
		return -1;
	}
	


}
