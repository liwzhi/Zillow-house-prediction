# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:49:49 2015

@author: weizhi
"""

import json
from pprint import pprint

with open('/Users/weizhi/Downloads/research-engineering-task-master/sampled_rc.json') as data_file:    
    data = json.loads(data_file)


with open('/Users/weizhi/Downloads/research-engineering-task-master/sampled_rc.json') as json_data:
    d = json.loads(json_data)
    json_data.close()
    pprint(d)
#pprint(data)
data = json.loads("/Users/weizhi/Downloads/research-engineering-task-master/sampled_rc.json")
import time
def process_change(wait_secs):
    start = time.time()
    print start
    while time.time() - start < wait_secs:
        time.sleep(0.001)
    print time.time()
process_change(2)