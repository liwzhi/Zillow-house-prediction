package workDay;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class testing {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		  long[] l1 = new long[]{30000,2000,50000,4000,10000,4500};
		  // generat container
		  RangeContainerFactory generateContainer = new RangeContainerFactory();
		  RangeContainer container = generateContainer.createContainer(l1);
		 
		  // get the PriorityQueue to hold the ids 
		 Ids idObj = container.findIdsInRange(3400, 4500, false, true);

          // get the ids one one
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
