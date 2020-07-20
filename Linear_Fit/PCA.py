import numpy as np
import matplotlib.pyplot as plt

class PCA:
    
    def __init__(self,freq):
        self.freq = freq
    
    #args path to data and frequency
    #returns data matrix and data dictionary
    def data_matrix(self,path):
        data = np.load('Data/'+path)
        d_matrix = data['arr_0'] #data matrix
        #for the y values iterate through number of pixels and get the column of intensity 
        #column of intensity corresponds to all intensity values of that certain freq
        data_dict = dict(zip(self.freq,(d_matrix[:,i] for i in range(d_matrix.shape[0]))))#x points keys and ypoints  values
        return d_matrix, data_dict
    
    #args matrix of data
    #returns covariance of data and normalized data set
    def cov_matrix(self,x_matrix):
        column_vec = x_matrix[:,np.arange(x_matrix.shape[1])] #taking each column vector of matrix
        mean_vector = np.c_[np.mean(column_vec,axis=0)] #calculating mean for each column and adding to vector
        ones_vector = np.ones([x_matrix.shape[0],1]) #one vector 
    #     print('Mean value ', np.dot(ones_vector,mean_vector.T), 'and mean vector \n', mean_vector)
        b_matrix = x_matrix - np.dot(ones_vector,mean_vector.T) #subtracting mean value 
        cov = (np.dot(b_matrix.T,b_matrix))/(x_matrix.shape[0]-1) #covariance matrix formula
        return cov,b_matrix 
    
    #arg matrix of data
    #returns correlation matrix
    def corr_matrix(self,x_matrix):
        c, stand = self.cov_matrix(x_matrix) #finding covariance of data matrix
        sigma = np.sqrt(np.diag(c)) #finding sigma vector from covariance 
        c /= sigma[:,None] #divide columns of c_matrix by sigma vector
        c /= sigma[None,:] #divide rows of c_matrix by sigma vector
        return c
    
    #args matrix
    #returns eigenvalues, eigenvectors, and tuple of eigvec and eigvec
    def eig_values(self,c):
        eigval,eigvec = np.linalg.eig(c) #finding eigenvalues and eigenvectors
        eig_pairs = [(eigval[i],eigvec[:,i]) for i in range(eigvec.shape[1])] #creating a tuple of eigval and eigvec
        eig_pairs.sort() #sorting from least to greatest
        eig_pairs.reverse() #reversing order to greatest to least
        return eigval, eigvec, eig_pairs
    
    #args eigenvalue
    #returns dictionary of ranked eigenvalues (keys number of rank and values explained variance of each eigenvalue)
    def ordered_eigval(self,e):
        total_eig = np.sum(e)
        var_exp = np.sort([(e[i]/total_eig) for i in np.arange(e.size)]) #calculating and sorting explained variance
        var_exp = var_exp[::-1] #reverse array to descending order
        e_dict = dict(zip(np.arange(1,var_exp.size+1),var_exp)) #adding ordered eigvalues to dictionary with rank as key
        return e_dict
    
    def graph_eigval(self,title, eig_dict):
        plt.figure()
        plt.scatter(eig_dict.keys(),eig_dict.values())
        plt.plot(list(eig_dict.keys()),list(eig_dict.values()))
        plt.yscale('log')
        plt.xlabel('Principal component number')
        plt.ylabel('Fraction of variance explained')
        plt.title(title)
        plt.show()