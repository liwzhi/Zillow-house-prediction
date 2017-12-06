package workdayDesign;

public class Node {
	public int data;
	public Node next = null;
	public Node prev = null;
	
	// specialized skip list pointers
	public Node nextHalf = null;
	public Node prevHalf = null;
	
	public Node nextQuarter = null;
	public Node prevQuarter = null;

}
