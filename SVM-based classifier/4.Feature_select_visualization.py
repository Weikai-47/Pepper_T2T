import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

data1 = pd.read_csv('SVM_list_line_linear.csv',header=None)[1:]
data2 = pd.read_csv('SVM_list_num_linear.csv',header=None)[1:]

plt.plot(data2,data1)

plt.xlabel('Number of Genes',fontsize=20) #x_label
plt.ylabel('ACC',fontsize=20)#y_label
# plt.title("The SVM Accuracy's Trend with the Increasing Numbers of Genes",fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()