package workdayDesign;

public class skipListEntry {
	public short key;
	public long value;
	
	public skipListEntry up;
	public skipListEntry down;
	public skipListEntry left;
	public skipListEntry right;
	
	public static short negInf = Short.MIN_VALUE;
	public static short posInf = Short.MAX_VALUE;
	
	skipListEntry(short keyValue,long val){
		key = keyValue;
		value = val;
	}
	
	

	

}
