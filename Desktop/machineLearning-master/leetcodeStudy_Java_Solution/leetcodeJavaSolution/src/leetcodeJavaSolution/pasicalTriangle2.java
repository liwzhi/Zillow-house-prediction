package leetcodeJavaSolution;

import java.util.ArrayList;
import java.util.List;

public class pasicalTriangle2 {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> result = new ArrayList<Integer>();
        List<Integer> pre = new ArrayList<Integer>();
        pre.add(1);
        if(rowIndex==0){
            return pre;
        }
   
        for(int i=1;i<=rowIndex;i++){
            List<Integer> curr = new ArrayList<Integer>();
            for(int j=0;j<=i;j++){
                if(j==0){
                    curr.add(1);
                }
                else if(j==i){
                    curr.add(1);
                }
                else{
                    curr.add(pre.get(j-1) + pre.get(j));
                }
            }
             pre = curr;
            
            
        }
        return pre;
        
        
    }
}
