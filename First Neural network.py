import numpy as np

#Add the matrix for transposed
the_matrix = np.matrix([[2, 5, 8, ],
                        [1, 7, 9, ],
                        [4, 8, 2, ],
                        [2, 5, 3, ]])

codifier = np.array([1, 4, 6, 7, ])

transposition = the_matrix.transpose() #Тransposed matrix

def sigmoid(sgm):
    return 1/(1 - np.exp(-sgm))  #Add sigmoid and expansion

np.random.seed(1)
sgm = 2 * np.random.random((4, 1)) *2 #Create a generation of new weights


#Output the data from the entire neural network
print("Matrix:")
print(the_matrix)
print("Codifier:")
print(codifier)
print("Transposed matrix:")
print(transposition)
print("Weight:")
print(sgm)