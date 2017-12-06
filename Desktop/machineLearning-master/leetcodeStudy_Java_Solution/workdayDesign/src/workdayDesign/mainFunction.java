package workdayDesign;

public class mainFunction {
	private static Node Head = null;
	private static Node Tail = null;
	private static Node Current = null;
	
	public static Node quarter = null;
	public static Node half = null;

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int ListCount = 100;
		for(int i=1;i<=ListCount;i++){
			Node newNode = new Node();
			newNode.data = i;
			
			if(Head ==null){
				Head = newNode;
				Tail = newNode;
				Current = newNode;
				quarter = newNode;
				half = newNode;
				
			}
			else{
				Current.next = newNode;
				newNode.prev = Current;
				if((i%25)==0){
					newNode.prevQuarter = quarter;
					quarter.nextQuarter = newNode;
					
					quarter = newNode;
				}
				if((i%50)==0){
					newNode.prevHalf = half;
					half.nextHalf = newNode;
					half = newNode;
				}
				
				Current = newNode;
				Tail = newNode;
				
				
			}
			
		}
		// search exmaples
		System.out.println("Find the number 7....");
		FindNumber(7);
	
		

	}
	
	// search in skiplist 
	public static void FindNumber(int num){
		Node currentNode = Head;
		boolean Found = false;
		
		
		while(currentNode!=null){
			if(currentNode.data>num){
				break;
			}
			
			if(currentNode.data!=num){
				if((currentNode.nextHalf!=null)&& (currentNode.nextHalf.data<=num)){
					currentNode = currentNode.nextHalf;
				}
				if((currentNode.nextQuarter!=null)&&(currentNode.nextQuarter.data<=num)){
					currentNode = currentNode.nextQuarter;
				}
				else{
					currentNode = currentNode.next;
				}
			}
			else{
				Found = true;
				break;
			}
			
			
			
			
			
		}
		if(Found) {System.out.println("number found");}
		else{System.out.println("Number wasn't found");}
		
		
	}
	
	
	
	

}
