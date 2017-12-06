package workDay;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class RangeContainer {
	 	
	Map<Short,Long> container;
	
	short bar = (short)0;
	// initial the RangeContainer 
	RangeContainer(Map<Short,Long> containerNew){
		container = containerNew;
	}
	
	
	// build the the PriorityQueue to hold the search result 
	 Ids findIdsInRange(long fromValue,long toValue,boolean fromInclusive,boolean toInclusive){
		 PriorityQueue<Short> heapq = new PriorityQueue<Short>();
		 for(Map.Entry<Short, Long> entry:container.entrySet()){
			 short id = entry.getKey();
			 long value = entry.getValue();
			 if(value==fromValue) {
				 if(fromInclusive){    // fromeInclusive is true, then push
				 heapq.add(id);}

			 }
			 else if(value ==toValue){
				 if(toInclusive){   // toInclusive is true, then push it to  
				 heapq.add(id);}
			 }
			 
			 else if(value>fromValue && value<toValue){
				 heapq.add(id);  // to get the ids
			 }
			 else{
				 continue; 
			 }	
			 
		 }
		 
		 Ids res = new Ids(heapq);
		 return res;
	 }
	
	

}
