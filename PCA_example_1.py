#usage: python3.10 PCA_example_1.py
"""
The Python version is 3.10.4.
This py script shows the procedure of SVD.

The script needs the following packages installed
numpy       1.21.6
matplotlib  3.5.2
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


data_list = []
data_len = 10

np.random.seed(100) #can change 100 to 101, 200, or 305, etc., to get different random numbers for np.random.randint
error_list = [np.random.randint(-30,30) for i in range(data_len)]
tmp_list = []

for i in range(data_len):
    tmp_list = []
    x = np.random.randint(-100,100) + 60
    #may change 2*x to 3*x or 4*x or -5*x    
    y = -2*x + error_list[i] + 50
    tmp_list.append(x)
    tmp_list.append(y)
    data_list.append(tmp_list)

data = np.array(data_list)

#means by coloumns to get mass center
mass_center = np.average(data, axis=0)

#move mass center to origin (0,0)
data_new =  np.zeros(np.shape(data))

for i in range(len(data)):
    data_new[i][0] = data[i][0] - mass_center[0]
    data_new[i][1] = data[i][1] - mass_center[1]

#get max for boundaries
max_index = np.abs(data).argmax()
max_element = data.flat[max_index]

ax1_x_limit = 2*abs(max_element)
ax1_y_limit = 2*abs(max_element)

ax2_x_limit = 4
#S_xx + S_yy is a constant for the data_new.
#axis=0, the column direction
ax2_y_limit = (np.sum(data_new**2,axis=0)[0] + np.sum(data_new**2,axis=0)[1])*1.2

ax3_x_limit = 4


plt.ion()  #interactive on
 
fig=plt.figure(num=1,figsize=(8,8))
gs=gridspec.GridSpec(3,2)
 
ax1=fig.add_subplot(gs[0:2,0:3])#centered data
ax1.set_xlim(-ax1_x_limit, ax1_x_limit)         
ax1.set_ylim(-ax1_y_limit, ax1_y_limit)

ax2=fig.add_subplot(gs[2:3,0:1])#S_xx,S_yy,S_xy
ax2.set_xlim(-ax2_x_limit, ax2_x_limit)


ax3=fig.add_subplot(gs[2:3,1:2])#S_xx/S_yy
ax3.set_xlim(-ax2_x_limit, ax2_x_limit)


#scatter
scatter_dot_size = 30
ax1.scatter(list(data[:,0]),list(data[:,1]),c='blue',s=scatter_dot_size)
plt.pause(1)


#moving data to center
move_steps = 90
pause_time = 0.03
tmp_arr = np.zeros(np.shape(data))

for i in range(1,move_steps+1):
    ax1.cla()
    ax1.set_xlim(-ax1_x_limit, ax1_x_limit)         
    ax1.set_ylim(-ax1_y_limit, ax1_y_limit)
    
    for j in range(len(data)):  
        tmp_arr[j][0] = data[j][0] - mass_center[0]/move_steps*i
        tmp_arr[j][1] = data[j][1] - mass_center[1]/move_steps*i
        
    ax1.scatter(list(tmp_arr[:,0]),list(tmp_arr[:,1]),c='blue',s=scatter_dot_size)
    plt.pause(pause_time)

plt.pause(2)

###rotation function. 1, anticlock; -1,clock
def rotate(degree_begin,degree_end,direction):
    S_xx = 0
    S_yy = 0    
    S_xx_list = []
    S_yy_list = []
    S_xy_list = []
    S_xx_S_yy_list = []
    F_S_xx_over_S_yy_list = []
    F_S_yy_over_S_xx_list = []
    SS_sequence = []

    delta_for = np.pi/180
    for i in range(degree_begin,degree_end,direction):
        ax1.cla()                               
        ax1.set_xlim(-ax1_x_limit, ax1_x_limit)         
        ax1.set_ylim(-ax1_y_limit, ax1_y_limit)

        ax2.cla()
        ax2.set_xlim(-ax2_x_limit, ax2_x_limit)
         
        ax3.cla()
        ax3.set_xlim(-ax2_x_limit, ax2_x_limit)

        tmp_arr = np.zeros(np.shape(data_new))
        tmp_arr = data_new @ np.array([[np.cos(i*delta_for),np.sin(i*delta_for)],
                                       [-np.sin(i*delta_for),np.cos(i*delta_for)]
                                      ])
        
        S_xx = np.sum(tmp_arr**2,axis=0)[0]
        S_yy = np.sum(tmp_arr**2,axis=0)[1]
        S_xy = np.dot(tmp_arr[:,0],tmp_arr[:,1])
        
        #for ax2    
        S_xx_list.append(S_xx)
        S_yy_list.append(S_yy)
        S_xy_list.append(S_xy)
        S_xx_S_yy_list.append(S_xx+S_yy)
                 
        #for ax3
        F_S_xx_over_S_yy_list.append(S_xx/S_yy)
        F_S_yy_over_S_xx_list.append(S_yy/S_xx)

        SS_sequence.append(i*delta_for)
        
        #draw    
        ax1.scatter(list(tmp_arr[:,0]),list(tmp_arr[:,1]),c='blue',s=scatter_dot_size)
        
        line1, = ax2.plot(SS_sequence,S_xx_list,c='blue',lw=0.5)
        line2, = ax2.plot(SS_sequence,S_yy_list,c='orange',lw=0.5)
        line3, = ax2.plot(SS_sequence,S_xy_list,c='darkorchid',lw=0.5)
        line4, = ax2.plot(SS_sequence,S_xx_S_yy_list,c='teal',lw=0.5)

        ax2.legend([line1,line2,line3,line4], ['S_xx','S_yy','S_xy','S_xx+S_yy'],loc='upper right',fontsize=6)

        ax3.plot(SS_sequence,F_S_xx_over_S_yy_list,label='S_xx/S_yy',c='blue',lw=0.5)
        ax3.plot(SS_sequence,F_S_yy_over_S_xx_list,label='S_yy/S_xx',c='orange',lw=0.5)
        ax3.legend(loc='upper right',fontsize=6)
        
        plt.pause(pause_time)




rotate(-90,90,1)

ax2.axhline(y=0,c='grey',ls='--',lw=1.5)


#solve equation (S_xx)'=0 with the Quadratics Formula
b_ = -(np.sum(data_new**2,axis=0)[0]-np.sum(data_new**2,axis=0)[1])/np.dot(data_new[:,0],data_new[:,1])

tan_d1 = (-b_ + math.sqrt(b_*b_ + 4))/2
tan_d2 = (-b_ - math.sqrt(b_*b_ + 4))/2

d1 = np.arctan(tan_d1)
d2 = np.arctan(tan_d2)

print("The rotation angle when S_xx is max or min.")
print(d1,"(equal to %s degrees)" % str(round(d1*180/np.pi,2)))
print(d2,"(equal to %s degrees)" % str(round(d2*180/np.pi,2)))
print('\n',end='')

#rotation matrix
rotation_matrix_right_side = np.array([[np.cos(d1),np.sin(d1)],
                                       [-np.sin(d1),np.cos(d1)]
                                      ])

S_xx = np.sum((data_new @ rotation_matrix_right_side)**2,axis=0)[0]
S_yy = np.sum((data_new @ rotation_matrix_right_side)**2,axis=0)[1]
S_xy = np.dot((data_new @ rotation_matrix_right_side)[:,0],(data_new @ rotation_matrix_right_side)[:,1])
print("The rotation matrix (right side) is")
print(rotation_matrix_right_side)
print('\n',end='')


print("The restult 2*2 matrix of S_xx, S_xy, S_yx, S_yy is")
print(S_xx,S_xy)
print(S_xy,S_yy)
print('\n',end='')

print("The rotation angle %s is the angle that maximize S_xx." % str(d1))
print("The rotation angle %s is the angle that minimize S_xx." % str(d2))
ax2.axvline(x=d1,c='blue',ls='--',lw=1)
ax2.axvline(x=d2,c='orange',ls='--',lw=1)
#ax3.axvline(x=d1,c='blue',ls='--',lw=1)
#ax3.axvline(x=d2,c='orange',ls='--',lw=1)

#solve equation (S_xy)'=0 with the Quadratics Formula
b_ = 4 * np.dot(data_new[:,0],data_new[:,1])/(np.sum(data_new**2,axis=0)[0]-np.sum(data_new**2,axis=0)[1])
                                            
tan_d1 = (-b_ + math.sqrt(b_*b_ + 4))/2
tan_d2 = (-b_ - math.sqrt(b_*b_ + 4))/2
                                              
d1 = np.arctan(tan_d1)
d2 = np.arctan(tan_d2)

print("The rotation angle when S_xy is max or min.")
print(d1)
print(d2)
print('\n',end='')

ax2.axvline(x=d1,c='darkorchid',ls='--',lw=1)
ax2.axvline(x=d2,c='darkorchid',ls='--',lw=1)

plt.pause(1)

plt.ioff()
plt.savefig('Fig_2')
plt.show()





"""
Now, find the rotation matrix with SVD
"""

"""
Get the eigenvalues and eigenvectors of $A^\mathrm{T}A$
np.linalg.eigh needs input of symetric matrix
Each column in eigenvectors is a eigenvector
"""
eigenvalues,eigenvectors = np.linalg.eigh(data_new.T@data_new)

#sort the eigenvectors by eigenvalues in reverse order
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

print("The eigenvalues of A^\mathrm{T}A")
print(eigenvalues)
print("The eigenvectors of A^\mathrm{T}A")
print(eigenvectors)

print('\n',end='')
print("The new basis vectors matrix")
V = eigenvectors
print(V)


