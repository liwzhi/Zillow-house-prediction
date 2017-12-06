class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        if not strs:
            return 
        listDict = []
        for string in strs:
            Dict = {}
            for item in string:
                if item in Dict:
                    Dict[item] +=1
                else:
                    Dict[item] =1
            listDict.append(Dict)
        result = [[listDict[0]]]
        result1 = [strs[0]]
        
        for index in range(1,len(listDict[1:])+1):
            Dict = listDict[index]
            flag = 0
            for i in range(len(result)):
                item = result[i][-1]
                if cmp(item,Dict) ==0:
                    result[i].append(item)
                    result1[i].append(strs[index])
                   # print 'hellp'
                    flag = 1
            if flag ==0:
                result.append([Dict])
                result1.append([strs[index]])
        return result1
                
obj = Solution()
strings = ["eat", "tea", "tan", "ate", "nat", "bat"]
result = obj.groupAnagrams(strings)    
print result        
        