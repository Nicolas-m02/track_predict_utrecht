#%%
import datetime
import numpy as np

t_stamp = 1.7804933298053448e+18

# convert nanoseconds timestamp to seconds
seconds = t_stamp / 1e9
dt_object = datetime.datetime.fromtimestamp(seconds)
print("Seconds:", seconds)
print("Datetime object:", dt_object)

print("Current time in seconds:", datetime.datetime.now().timestamp())
print("Latency in seconds:", datetime.datetime.now().timestamp() - seconds)
