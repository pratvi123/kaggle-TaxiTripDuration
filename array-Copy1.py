
# coding: utf-8

# In[48]:


import numpy as np


# In[49]:


num_array = list()


# In[50]:


num=raw_input("enter the no of elements")
print 'Enter numbers in array: '
for i in range(int(num)):
    n = raw_input("num:")
    num_array.append(int(n))
key=raw_input("enter the search key")
for i in num_array:
    if(i==key):
        print 'key is found in position',num_array.index(i)


# num=raw_input("enter the no of elements")
# print 'Enter numbers in array: '
# for i in range(int(num)):
#     n = raw_input("num:")
#     num_array.append(int(n))

# In[35]:


# num_arr=list()
# n=raw_input("enter the no of elements")
# print 'Enter numbers in array: '
# for i in range(int(n)):
#     n = raw_input("num:")
#     num_arr.append(int(n))


# In[37]:


# num_arr.sort()
# num_arr


# In[38]:


# num_arr.reverse()


# In[39]:


# num_arr


# In[44]:


# num_array=[1,2,3]
# k=3
# for i in num_array:
#     if(i==k):
#         print 'key is found in position',num_array.index(i)
    

