package workDay;

import java.util.HashMap;
import java.util.Map;

public class RangeContainerFactory {
	
	private Map<Short,Long> cons = new HashMap<Short,Long>();
	
	
	RangeContainer createContainer(long[] data) {
		 
		 for(Long item:data){
			 cons.put((short) cons.size(), item);
		 }

		 RangeContainer range = new RangeContainer(cons);

		return range;
		 
	 }
}
