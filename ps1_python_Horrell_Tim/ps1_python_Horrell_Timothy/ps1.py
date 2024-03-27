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

#define vector y using where function
y = z[np.where((z >= 0) & (z < 1.5))]

#print the length of vector y
lenY = len(y)
print("Elements copied to y: " + str(lenY))

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
rowSumA = np.sum(A, axis=1)
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

##note in report what I should get and what I got, couldn't figure out with a quick debug

#calculate norms
x1_l1Norm = np.linalg.norm(x1, ord = 1, axis = 1)
x2_l1Norm = np.linalg.norm(x2, ord = 1, axis = 1)
x1_l2Norm = np.linalg.norm(x1)
x2_l2Norm = np.linalg.norm(x2)

#print norms
print("x1, l1 norm: " + str(x1_l1Norm) + "\nx2, l1 norm: " + str(x2_l1Norm) + "\nx1, l2 norm: " + str(x1_l2Norm) + "\nx2, l2 norm: " + str(x2_l2Norm))

#######################################################
#Number 5
#######################################################

#function to return a vector, of squared row values
def sum_Sq_ROW(A):
    #get columns in A and make an empty vector
    size = np.shape(A)
    B = np.zeros(size[1])

    #square A
    Asq = np.square(A)

    #print the input matrix
    print("The input array is" + str(A))

    #fill vector B with the sum
    for n in range(0, size[1]):
        B[n] = np.sum(Asq, axis=0)[n]
    
    return B

Array1 = np.array([[1, 2, 3], [4, 5, 6]])
Array2 = np.array([[2, 2], [3, 3], [4, 4], [5, 5]])

print("The sum squared vector 1 is: " + str(sum_Sq_ROW(Array1)))
print("The sum squared vector 2 is: " + str(sum_Sq_ROW(Array2)))