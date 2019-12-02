#!/usr/bin/env python
# coding: utf-8

# Install packages we'll use in this note book

# In[1]:


get_ipython().run_line_magic('pip', 'install scikit-learn pandas matplotlib')


# **Restart the kernel for the installations to take effect**

# Set up the Intelligent Plant clients.

# In[1]:


import intelligent_plant.app_store_client as app_store_client
from os.path import expanduser
app_store = app_store_client.AppStoreClient(open(expanduser("~/.access_token"), "r").read())
data_core = app_store.get_data_core_client()


# In[2]:


import intelligent_plant.utility as utility
import matplotlib.pyplot as plt


# Import the MLP regressor from Scikit-learn
# 
# Make sure you restart the kernel after installing

# In[3]:


from sklearn.neural_network import MLPRegressor


# We'll query all of the tags in IP Datasource 2 and make a list of just their names

# In[15]:


dsn = "FCBB05262EADC0B147746EE6DFB2B3EA5C272C33C2C5E3FE8F473D85529461CA.Edge Historian"


# In[116]:


#tags = data_core.get_tags(dsn, 1, 113)
tags=['mqtt.intelligentplant.com::BoilerRoomTemp[1].DegC.','mqtt.intelligentplant.com::BoilerRoomTemp[2].DegC.','mqtt.intelligentplant.com::BoilerRoomTemp[3].DegC.','mqtt.intelligentplant.com::BoilerRoomTemp[0].DegC.']
tag_names = tags


# In[117]:


tag_names


# In[118]:


all_data = data_core.get_processed_data({dsn: tag_names}, "*-2d", "*", "1m","AVG")


# Convert the returned data into a data frame to make it easier to work with

# In[119]:


all_data_frame = utility.query_result_to_data_frame(all_data)


# In[120]:


all_data_frame


# In[121]:


all_data_frame.plot(x="TimeStamp", legend=False)
plt.show()


# To make the data easier to process we need to normalise it

# In[112]:


import sklearn.preprocessing
from pandas import DataFrame


# In[113]:


scaler = sklearn.preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(all_data_frame.loc[:, all_data_frame.columns != 'TimeStamp'])
normalised_df = DataFrame(scaled_values)


# In[114]:


normalised_df


# In[115]:


normalised_df[:10].plot(legend=False)


# In[ ]:


training_input = normalised_df.iloc[:-10,:-1]
training_output = normalised_df.iloc[:-10,-1]

testing_input = normalised_df.iloc[-10:,:-1]
testing_output = normalised_df.iloc[-10:,-1]


# In[ ]:


plt.plot(training_input[:10])


# The MLP is going to try and learn the relationship between the graph above and below

# In[ ]:


plt.plot(training_output[:10])


# In[ ]:


mlp = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)


# In[ ]:


mlp.fit(training_input, training_output)


# In[ ]:


prediction = mlp.predict(testing_input)


# In[ ]:


prediction_df = DataFrame({ "actual": testing_output, "prediction": prediction })


# In[ ]:


prediction_df.plot()


# In[ ]:




