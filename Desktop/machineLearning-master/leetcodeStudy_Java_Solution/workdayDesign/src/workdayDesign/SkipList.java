package workdayDesign;

import java.util.Random;

public class SkipList {
	
	public skipListEntry head;
	public skipListEntry tail;
	
	public int n;  // number of entires in the skip list
	
	public int h; // Height;
	
	public Random r; // coin toss
	
	
	
	public SkipList(){
	//	skipListEntry p1,p2;
		
		skipListEntry p1 = new skipListEntry(skipListEntry.negInf,(Long) null);
		
		skipListEntry p2 = new skipListEntry(skipListEntry.negInf,(Long) null);		
		
		// link the 
		p1.right = p2;
		p2.left = p1;
		
		
		head = p1;
		
		tail = p2;
		
	}
	

}
