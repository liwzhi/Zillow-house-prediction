package workDay;

import java.util.PriorityQueue;

public class Ids {
	PriorityQueue<Short> heapq;
	
	Ids(PriorityQueue<Short> h){
		heapq = h;
	}
	
	// get the id in sequence
	short nextId(){
		
		if(!heapq.isEmpty()){
			 return heapq.poll();
		}
		
		return -1;
	}

}
