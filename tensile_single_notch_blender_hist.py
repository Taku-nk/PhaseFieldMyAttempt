# Implements the fourth (seocnd?)-order phase field to study the growth of fracture in a two dimensional plate
# The plate has initial crack and is under tensile loading
# Load the data file for the final refined domain to obtain the crack path

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import scipy.io
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# tf.logging.set_verbosity(tf.logging.ERROR)


import sys
sys.path.append(
    "C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadAnsysResult")
from sample_result import ResultSampler


from utils.PINN2D_PF_Blender import CalculateUPhiBlender

np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_PF(CalculateUPhiBlender):
    '''
    Class including (symmetry) boundary conditions for the tension plate
    '''
    def __init__(self, model, NN_param):
        
        super().__init__(model, NN_param)
        
    def net_uv(self,x,y,vdelta):

        X = tf.concat([x,y],1)

        uvphi = self.neural_net(X,self.weights,self.biases)
        uNN = uvphi[:,0:1]
        vNN = uvphi[:,1:2]
        
        u = (1-x)*x*uNN
        v = y*(y-1)*vNN + y*vdelta

        return u, v
    
    def net_hist(self,x,y):
        """returns initial history (dimension of positive strain energy density) function """
        shape = tf.shape(x)
        self.crackTip = 0.5
        init_hist = tf.zeros((shape[0],shape[1]), dtype = np.float32)
        dist = tf.where(x > self.crackTip, tf.sqrt((x-0.5)**2 + (y-0.5)**2), tf.abs(y-0.5))
        init_hist = tf.where(dist < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist/self.l))/self.l, init_hist)
        
        return init_hist
    
    def net_energy(self,x,y,hist,vdelta):
        """ hist = hist_f in the source. (maximum history positive strain energy density)"""
        u, v = self.net_uv(x,y,vdelta)
        phi = self.net_phi(x,y)
        
        g = (1-phi)**2
        phi_x = tf.gradients(phi, x)[0]
        phi_y = tf.gradients(phi, y)[0]
        nabla = phi_x**2 + phi_y**2
        
        u_x = tf.gradients(u,x)[0]
        v_y = tf.gradients(v,y)[0]
        u_y = tf.gradients(u,y)[0]
        v_x = tf.gradients(v,x)[0]
        u_xy = (u_y + v_x)
        
        hist = self.net_update_hist(x, y, u_x, v_y, u_xy, hist) 
        
        sigmaX = self.c11*u_x + self.c12*v_y
        sigmaY = self.c21*u_x + self.c22*v_y
        tauXY = self.c33*u_xy
        
        energy_u = 0.5*g*(sigmaX*u_x + sigmaY*v_y + tauXY*u_xy)
        energy_phi = 0.5*self.cEnerg * (phi**2/self.l + self.l*nabla) + g* hist
        
        return energy_u, energy_phi, hist


def plot_save_loss_convergence(adam_buff, lbfgs_buff, save_path='', plot=False):
    """save losses to one csv file
    Args:
        loss_list: list. [adam_buff, lbfgs_buff]
        save_path: path like. .csv and .svg will be saved.
        plot: bool. When plot==False, no visualization happen (but still saves the figure)
    Returns:
    """


    adam_iter = np.arange(1, len(adam_buff)+1)
    adam_loss_arr= np.array(adam_buff)
    adam_loss_type = np.full_like(adam_loss_arr, 0) # 0 for adam, 1 for lbfgs 

    lbfgs_iter = np.arange(len(adam_buff)+1, len(adam_buff)+len(lbfgs_buff)+1)
    lbfgs_loss_arr = np.array(lbfgs_buff)
    lbfgs_loss_type = np.full_like(lbfgs_loss_arr, 1)
    
    it = np.concatenate((adam_iter, lbfgs_iter), axis=0)
    loss =np.concatenate((adam_loss_arr, lbfgs_loss_arr), axis=0) 
    loss_type = np.concatenate((adam_loss_type, lbfgs_loss_type), axis=0)


    loss_df = pd.DataFrame({'i':it, 'loss':loss, 'id':loss_type})
    loss_df.to_csv(save_path, index=False)
    





        






if __name__ == "__main__":
    
    model_name = 'test_notch_blender_hist'
    model_dir = Path(f'./{model_name}')
    output_dir = model_dir/'output'
    input_dir = model_dir/'input'
    output_dir.mkdir(parents=False, exist_ok=True)

    
    nSteps = 2 # Total number of steps to observe the growth of crack 
    deltaV = 1e-3 # Displacement increment per step  
    
    model = dict()
    model['E'] = 210.0*1e3
    model['nu'] = 0.3
    model['L'] = 1.0
    model['W'] = 1.0
    model['l'] = 0.0125 # length scale parameter
    model['cEnerg'] = 2.7 # critical energy release rate
    model['B'] = 92 # for hist
    
    
    # Domain bounds
    model['lb'] = np.array([0.0,0.0]) #Lower bound of the plate
    model['ub'] = np.array([model['L'],model['W']]) # Upper bound of the plate <- this is used for normalization of input 

    NN_param = dict()
    NN_param['layers'] = [2, 50, 50, 50, 3]
    NN_param['data_type'] = tf.float32
       
    
    # data = resd_csv()
    #---------------------------------------------------------------------------
    # Read Input Data
    #---------------------------------------------------------------------------
    X_f = pd.read_csv(input_dir/'x_internal.csv').loc[:, ['xc','yc','area']].to_numpy()# X_f[:, 0:1]=xs, X_f[:, 1:2]=ys, X_f=[:, 2:3]
    # initial history boundary (maximum tensile strain energy density)     
    model['hist_init'] = model['B']*model['cEnerg']*0.5/model['l'] * (pd.read_csv(input_dir/'x_hist_n.csv').loc[:, ['hist_n']].to_numpy())
    
    

       
    Grid =pd.read_csv(input_dir/'x_validation.csv').loc[:, ['xc','yc']].to_numpy()# Grid[:, 0] = x, Grid[:, 1] = y
    # history H place holder for validation (maximum tensile strain energy density)     
    hist_grid = np.transpose(np.array([np.zeros((Grid.shape[0]),dtype = np.float32)]))
    #---------------------------------------------------------------------------

    phi_pred_old = hist_grid #Initializing phi_pred_old to zero # this is only for validation not for training
    
    modelNN = PINN_PF(model, NN_param)
    # just a placeholde it will be replaced by maximum(init, zeros)
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
    num_train_its = 15000
    # num_train_its = 150



    
    for iStep in range(0,nSteps):
        iter_output_dir = output_dir/f'iter_{iStep+1}'
        iter_output_dir.mkdir(parents=False, exist_ok=True)
                
        v_delta = deltaV*iStep                
        
        
        if iStep==0:
            # num_lbfgs_its = 100
            num_lbfgs_its = 10000
        else:
            # num_lbfgs_its = 10
            num_lbfgs_its = 1000

        start_time = time.time()                    
        modelNN.train(X_f, v_delta, hist_f, num_train_its, num_lbfgs_its)
        
        _, _, phi_f, _, _, hist_f = modelNN.predict(X_f[:,0:2], hist_f, v_delta) # Computing the history function for the next step        
                              
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        # You can naively put hist_grid (initial zero) because, in predict, 
        # hist_grid is compared using tf.maximum and relaced with appropriate values.
        u_pred, v_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = modelNN.predict(Grid, hist_grid, v_delta)
        
        phi_pred = np.maximum(phi_pred, phi_pred_old) # has to increase. No spontaneous mending.
        phi_pred_old = phi_pred

        # ----------------------------------------------------------------------
        # Predict and save result
        # ----------------------------------------------------------------------
        sampler = ResultSampler()
        sampler.load_numpy_result(
            Grid[:, 0], Grid[:, 1],
            value_dict={
                'disp_x': u_pred.flatten(),
                'disp_y': v_pred.flatten(),
                'phi': phi_pred.flatten(),
                'elastic_energy_density': elas_energy_pred.flatten(),
                'fracture_energy_density': frac_energy_pred.flatten(),
                'hist': hist_grid.flatten()
                # 'stress_x': sigma_x_pred.flatten(),
                # 'stress_y': sigma_y_pred.flatten(),
                # 'stress_xy': sigma_xy_pred.flatten(),
            }
        )
        sampler.save_original(iter_output_dir/'DEM_original_result.csv')

        # save loss
        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = modelNN.lbfgs_buffer
        plot_save_loss_convergence(adam_buff, lbfgs_buff, save_path=iter_output_dir/'loss.csv', plot=True)
        

        
        # ----------------------------------------------------------------------
        # 1D plot of phase field
        # ----------------------------------------------------------------------
        xVal = 0.25
        nPredY = 2000
        xPred = xVal*np.ones((nPredY,1))
        yPred = np.linspace(0,model['W'],nPredY)[np.newaxis]
        xyPred = np.concatenate((xPred,yPred.T),axis=1)
        phi_pred_1d = modelNN.predict_phi(xyPred)
        phi_exact = np.exp(-np.absolute(yPred-0.5)/model['l'])
        
        plt.plot(yPred.flatten(), phi_pred_1d.flatten(), c='tab:blue', ls='-', label=f'DEM iter={iStep+1}')
        plt.plot(yPred.flatten(), phi_exact.flatten(), c='tab:orange', ls=':', label='Exact')
        plt.legend()
        plt.savefig(iter_output_dir/'comparison_1d.svg', format='svg')
        plt.cla()
        # plt.show()
        
        error_phi = (np.linalg.norm(phi_exact-phi_pred_1d,2)/np.linalg.norm(phi_exact,2))
        print('Relative error phi: %e' % (error_phi))
        
        print('Completed '+ str(iStep+1) +' of '+str(nSteps)+'.')    

