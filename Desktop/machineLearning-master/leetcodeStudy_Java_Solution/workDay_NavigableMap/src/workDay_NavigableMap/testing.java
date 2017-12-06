package workDay_NavigableMap;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


public class testing {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		  long[] l1 = new long[]{30000,2000,50000,4000,10000,4500};

		
		  // split the whole data into different bukect, it is from the minimum value to maximum value
		 // for example: {0=2000, 1=4000, 2=6000, 3=8000, 4=10000, 5=12000, 6=14000,...}
		 
		  long diff = 2000;
		  RangeContainerFactory contauberObj = new RangeContainerFactory(diff);
		  Map<Integer, ListNode> container = contauberObj.createContainer(l1); // build the container
		//  System.out.println(contauberObj.keyToIncomeMap); // get the key --> income, bucket 
		  
		//  System.out.println(container.keySet());; // check the non-empty bucket		  		  
		  
		  RangeContainer rangeObj = new RangeContainer(container,contauberObj.Keys); // get the object
		  
		  PriorityQueue<Short> res = rangeObj.findIdsInRange(3400, 4500, false, true); // get the ids n priority queue. 
		  System.out.print("the output of id number is: ");
		  System.out.println(res.size());

		 // PriorityQueue --> pop ids in sequence
		  
		  Ids idObj = new Ids(res);   // get the Ids object
		  
		  
		  
		  short index = idObj.nextId();   // get the value
		  List<Short> idList = new ArrayList<Short>();
		  List<Long> outputList = new ArrayList<Long>();

		  while(index!=-1){;
		  
			  idList.add(index);
			  outputList.add(l1[index]);
			  index = idObj.nextId();
			  
		  }
		  
		  System.out.print("the ids is: ");
		  System.out.println(idList);

		  System.out.print("outputs value is: ");
		  System.out.print(outputList);


	}

}
