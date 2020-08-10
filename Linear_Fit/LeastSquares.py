import numpy as np

class LeastSquares:
    
    def __init__(self,data):
        self.data = data

    def lin_matrix(self):
        one_arr = np.ones([self.data.size,1]) #array of ones concatenated to matrices
        lin = np.hstack([one_arr, self.data]) #appending array of ones to array of data matrix
        return lin
    
    def quad_matrix(self):
        lin = self.lin_matrix()
        quad = np.hstack([lin,self.data**2]) #appending lin to array of data squared 
        return quad

    def cub_matrix(self):
        quad = self.quad_matrix()
        cub = np.hstack([quad,self.data**3]) ##appending quad to array of data cubed
        return cub
                         
    def pow_matrix(self):
        one_arr = np.ones([self.data.size,1])
        pow_m = np.hstack([one_arr,np.log(self.data)]) #appending array of ones to array of log xvalues
        return pow_m

    #parameters data_matrix, y points, and noise covariance
    #returns y_model, x_bar (parameters for fit) and error covariance
    def ymodel(self,data_matrix,yval,noise_cov=None):
        if noise_cov == None:
            noise_cov = np.identity(data_matrix.shape[0])
                        
        #calculating parameters
        dot_matrix = np.dot(data_matrix.T,np.linalg.inv(noise_cov)) #Step 1
        doty_matrix = np.dot(dot_matrix,yval) #Step 2
        inv_matrix = np.linalg.inv(np.dot(dot_matrix,data_matrix)) #Step 3
        x_bar = np.dot(inv_matrix, doty_matrix) #Step 4
        predict_y = np.dot(data_matrix, x_bar) #Step 5
        return predict_y, x_bar, inv_matrix

    def coef(self,data_matrix,yval,noise_cov=None):
        predict_y,x_bar,inv_matrix = self.ymodel(data_matrix,yval)
        return x_bar
             
    #args extended data_matrix (with extra x points added), y points, index where actual data begins in data_matrix_extend
    #returns y_model, x_bar (parameters for fit) and error covariance                   
    def ymodel_extend(self,data_matrix_extend,yval,index_begin_x):
        data_matrix = data_matrix_extend[index_begin_x:] #slicing array to section with only x values from data
        predict_y,x_bar,inv_matrix = self.ymodel(data_matrix,yval)
        predict_y = np.dot(data_matrix_extend, x_bar)
        return predict_y,x_bar,inv_matrix

    def error_bar(self,err_cov):
        err = np.sqrt(np.diag(err_cov)) #taking the square root of the diagonal of V
        return err