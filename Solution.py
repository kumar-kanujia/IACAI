#!/usr/bin/env python
# coding: utf-8

# In[33]:


from PIL import Image
import os
import numpy as np
import random
import tensorflow as tf

model = tf.keras.models.load_model('saved_model/my_model')

test=[]
test_dir_files = os.listdir('test')
for img_name in  test_dir_files:
    img=Image.open('test/'+img_name)
    img=np.array(img)
    test.append(img)
test=np.array(test)/255

result=[int(x) for x in list(np.argmax(model.predict(test),axis=1))]


# In[35]:


import random
import json
def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data

def generate_sample_file(filename):
    res = {}
    for i in range(len(result)):
        test_set = test_dir_files[i]
        res[test_set] = result[i]
    #return res
    write_json(filename, res)

if __name__ == '__main__':
    generate_sample_file('./result.json')


# In[ ]:




