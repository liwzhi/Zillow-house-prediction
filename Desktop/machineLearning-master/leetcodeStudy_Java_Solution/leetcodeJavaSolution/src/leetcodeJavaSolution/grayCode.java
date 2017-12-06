package leetcodeJavaSolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class grayCode {
	public List<Integer> grayCode(int n){
		List<Integer> result = new ArrayList<Integer>();
		
		
		
		
		return result;
		
	}
	
	public List<Integer> helper(String bits,List<Integer> result,Stack<String> stk, int start,List<Boolean> flag){
		if(stk.size()== bits.length()){
			String res ="";
			while(!stk.isEmpty()){
				res+=stk.pop();
			}
			
			result.add(Integer.parseInt(res));
			return result;
		}
		for(int i=start;i<bits.length();i++){
			if (flag.get(i)==false){
				stk.push(bits.substring(i, i+1));
				flag.set(i, true);
				helper(bits,result,stk,i+1,flag);
				stk.pop();
				flag.set(i,false );	
			}	
		}	
		return result;
		
	}

}
