import pickle
import os

f=[]
data_list=[]
n=0
log_path="games\\arkanoid\\log\\"
for filename in os.listdir(log_path):
    with open(log_path+filename, "rb") as fi:
        f.append(fi)
        data_listi = pickle.load(fi)
        data_list.append(data_listi)
    n=n+1
N=n-1

# save each info separetely
Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
for n in range(0, N):
    for i in range(0, len(data_list[n])):
        Frame.append(data_list[n][i].frame)
        Status.append(data_list[n][i].status)
        Ballposition.append(data_list[n][i].ball)
        PlatformPosition.append(data_list[n][i].platform)
        Bricks.append(data_list[n][i].bricks)

#____________________________________________________________________________________________________________
import numpy as np
PlatX_plus_20 = []
PlatX = np.array(PlatformPosition)[:,0][:,np.newaxis]
for i in range(0,len(PlatX)):
    PlatX_plus_20.append(20)
PlatX_20=np.array(PlatX_plus_20)[:,np.newaxis]
PlatX = PlatX + PlatX_20
PlatX_next=PlatX[1:,:]
instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

#球移動方向
BallX_position = np.array(Ballposition)[:,0][:,np.newaxis]
BallX_position_next = BallX_position[1:,:]
Ball_Vx = BallX_position_next - BallX_position[0:len(BallX_position_next),0][:,np.newaxis]
BallY_position = np.array(Ballposition)[:,1][:,np.newaxis]
BallY_position_next = BallY_position[1:,:]
Ball_Vy = BallY_position_next - BallY_position[0:len(BallY_position_next),0][:,np.newaxis]

#Select some features to make x
#x為輸入特徵向量
Ballarray=np.array(Ballposition[:-1])
x=np.hstack((Ballarray,PlatX[0:-1,0][:,np.newaxis],Ball_Vx,Ball_Vy))
#Select instructions as y
y=instruct

#split train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.1, random_state=1200)

#%%train your model here
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
svr=SVR(kernel='rbf',gamma='scale' ,degree=1,C=3.0,epsilon=0.001)
svr.fit(x_train,y_train)

y_predict=svr.predict(x_test)
print(y_predict)
#acc_arc_bef_scaler=accuracy_score(yp_bef_scaler,y_test)

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_predict, y_test)
RMSE=np.sqrt(MSE)
print(MSE)
print(RMSE)

filename1="C:\\Users\\BBS\\Desktop\\Machine leaning\\MLGame-master\\games\\arkanoid\\ml\\svr_test.sav"
pickle.dump(svr, open(filename1, 'wb'))