import numpy as np
import datetime
# date,open,high,low,close,volume


def strTodate(str1):
    return datetime.datetime.strptime(str1.decode('ascii'), "%Y-%m-%d").date().weekday()
# date,open,high,low,close,volume


d, v = np.loadtxt('eye.txt', delimiter=',', skiprows=1, usecols=(0,4), unpack=True, converters={0:strTodate})
for i in range(0,5):
    indices = np.where(d==i)
    price = np.take(v, indices)
    avg = np.mean(price)
    print("星期",i+1,"股价是",price,"平均股价是", avg)