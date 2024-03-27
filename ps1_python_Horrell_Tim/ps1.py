import numpy as np
import matplotlib.pyplot as plt
import time

#######################################################
#Number 3
#######################################################

#fill x and z with random numbers of various distributions
x = np.random.normal(loc=1.5, scale=0.6, size=1000000)
z = np.random.uniform(-1, 3, 1000000)

#define the figure and axes objects for vector x
fig, axs = plt.subplots()

#define attributes and save the histogram
axs.hist(x, label = "vector x", density = True)
plt.title("vector x")
plt.savefig("ps1-3-c-1.png")

#redefine fig and axs in plt
fig, axs = plt.subplots()

#build and save the vector z plot
axs.hist(z, label = "vector z", density = True)
plt.title("vector z")
plt.savefig("ps1-3-c-2.png")

#get length of vector x, this is how many times to loop and print
xLen = len(x)
print("Length of Vector x is: " + str(xLen))

#start the timer and loop
loopStart = time.time()
for i in range(0, xLen):
    x[i] += 1
loopEnd = time.time()

#time of operation
loopTime = loopEnd - loopStart
print("Loop Add Time: " + str(loopTime))

#add one to each element without looping
addTimeStart = time.time()
x = x + 1
addTimeEnd = time.time()
addTime = addTimeEnd - addTimeStart
print("Non-loop Add Time: " + str(addTime))

#define vector y. HOW TO LIMIT RUNTIME IN NP.ARRAYS??
'''
y = np.array([])
for i in range(0, len(z)):
    if(z[i] >= 0 and z[i] < 1.5):
        y = np.append(y, z[i])

lenY = len(y)
print("Elements copied to y: " + str(lenY))
'''

#######################################################
#Number 4
#######################################################

#define matrix A and find certain attributes
A = np.array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
maxRow1 = np.max(A[0])
maxRow2 = np.max(A[1])
maxRow3 = np.max(A[2])
minCol1 = np.min(A[:,0])
minCol2 = np.min(A[:,1])
minCol3 = np.min(A[:,2])
minValA = np.min(A)
rowSumA = A[0] + A[1] + A[2]
totSumA = np.sum(A)

#compute matrix B
B = np.square(A)

#system of linear equations
#Coefficient Matrix
Coef = np.array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])

#Constant Matrix
Const = np.array([[1], [3], [5]])

#Solution Matrix
Sol = np.linalg.solve(Coef, Const)

print("x equals: " + str(Sol[0]) + "\ny equals: " + str(Sol[1]) + "\nz equals: " + str(Sol[2]))

#initialize vectors x1 and x2
x1 = np.array([[0.5, 0, -1.5]])
x2 = np.array([[1, -1, 0]])

#calculate norms
x1_l1Norm = np.linalg.norm(x1, ord = 1)
x2_l1Norm = np.linalg.norm(x2, ord = 1)
x1_l2Norm = np.linalg.norm(x1)
x2_l2Norm = np.linalg.norm(x2)

#print norms
print("x1, l1 norm: " + str(x1_l1Norm) + "\nx2, l1 norm: " + str(x2_l1Norm) + "\nx1, l2 norm: " + str(x1_l2Norm) + "\nx2, l2 norm: " + str(x2_l2Norm))

#######################################################
#Number 5
#######################################################

def sum_Sq_ROW(A):
    size = np.shape(A)
    np.array.
