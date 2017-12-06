package leetcodeJavaSolution;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

public class LCA_BST {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
	if(root==null)
			return null;
		if(root.val>p.val && root.val>q.val){
			return lowestCommonAncestor(root.left,p,q);}
		else if(root.val<p.val && root.val<q.val){
			return lowestCommonAncestor(root.right,p,q);}
		else {
			return root;}
	}
    
    
    
    public TreeNode LCA(TreeNode root,TreeNode p, TreeNode q){
    	HashMap<TreeNode,TreeNode> Dict = new HashMap<TreeNode,TreeNode>();
    	Stack<TreeNode> stk = new Stack<TreeNode>();
    	stk.push(root);
    	Dict.put(root, null);
    	while(!Dict.containsKey(p) || !Dict.containsKey(q)){
    		TreeNode item = stk.pop();
    		if(item.left!=null){
    			stk.push(item.left);
    			Dict.put(item.left,item);
    		}
    		if(item.right!=null){
    			stk.push(item.right);
    			Dict.put(item.right, item);
    		}

    	}
    	
    	Set<TreeNode> path = new HashSet<TreeNode>();
    	while(p!=null){
    		path.add(p);
    		p = Dict.get(p);
    		
    	}
    	while(!path.contains(q)){
    		q = Dict.get(q);
    	}
    	return q;
	
    }
}
