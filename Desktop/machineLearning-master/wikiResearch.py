# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:42:36 2015

@author: weizhi
"""

import json
import time
data = []

with open('/Users/weizhi/Downloads/research-engineering-task-master/sampled_rc.json') as f:
    for line in f:
        data.append(json.loads(line))


def process_change(wait_secs):
    start = time.time()
    while time.time() - start < wait_secs:
        time.sleep(0.001)
#%% data generate
import numpy as np
processingTime = np.random.uniform(0.1,10,len(data))