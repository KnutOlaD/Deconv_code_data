# -*- coding: utf-8 -*-
"""
Created July 2021
Version 1.56
@author: Knut Ola Dølven and Juha Vierinen

Contains deconvolution, inlcuding automatic Delta T selection as described in 
Dølven et al., (2022), but also includes a variant using Tikhonov regularization and all functions 
producing the results presented in Dølven et al., (2021). 
The function that does deconvolution with automatic delta_t selection as  described in 
Dølven et al., (2022) is named "deconv_master() which integrates all functions and is 
the final product of this work. 
Supporting functions are:
    
    diffusion_theory - Builds the theory matrix (m=Gx or eq.5 and 6 in Dølven et al., (2022))
    
    test_ua - Function that creates a step-change signal
    
    forward_model - Convolution model to simulate sensor response/measurements from an arbiry signal
    
    sime_meas - Function that makes simulated measurements using forward_model and test_ua
    
    find_kink - Regularization optimization using L-curve analysis finding maximum gradient point
    
    estimate_concentration - Estimates concentration using a least squares solution of diffusion_theory 
        and measurement/setting input. Switch to non-negative least squares can be done here. 
        
    field_example_tikhonov - Estimates concentration using the field example data (see Dølven et al., 2022)
        and tikhonov regularization
        
    field_example_model_complexity - Estimates concentration using field example data and complexity
        regularization
        
    deconv_master - Estimates concentration for any dataset/input parameters (described in help(deconv_master))
            returns vectors containing the estimated corrected data, measurement estimate, model time, 
            standard deviation *2 (95% confidence) . The function also provides an L-curve plot, fit residual 
            plot and estimate plot.
                            
    
    """

import matplotlib
SMALL_SIZE = 14
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)
import numpy as n
import matplotlib.pyplot as plt
import scipy.signal as s #used if you want to use nonzero 
import scipy.optimize as so
import scipy.io as sio
import scipy.interpolate as sint
from scipy.interpolate import UnivariateSpline as unisp

#Diffusion_theory Function that builds the theory matrix
def diffusion_theory(u_m,                           # these are the measurements
                     t_meas,                        # measurement times
                     missing_idx=[],                # if measurements are missing
                     k=1.0,                         # diffusion coefficient
                     t_model=n.linspace(0,5,num=100),   # time for model
                     sigma=0.01,                    # u_m measurement noise standard deviation
                     L_factor=1e5,
                     smoothness=1.0):                   

    #Vector/matrix sizes:
    n_meas=len(t_meas)
    n_model = len(t_model)
    #Warning trigger
    if n_model > 2000:
        print("You are using a lot of model points. In order to be able to solve this problem efficiently, the number of model points should be <2000")
    
    #Size of G-matrix
    A = n.zeros([n_meas + n_model + n_model-2,n_model*2])
    
    #Size of measurement matrix
    m = n.zeros(n_meas + n_model + n_model-2)
    dt = n.diff(t_model)[0]
    
    # diffusion equation
    # L is a very large number to ensure that the differential equation solution is nearly exact
    # this assentially means that these rows with L are equal to zero with a very very variance.
    # L >> 2*dt*1.0/n.min(sigma)
    L=2*dt*L_factor/n.min(sigma)
    print(L)
    for i in range(n_model):
        m[i]=0.0

        # these two lines are k(u_a(t) - u_m(t)) i.e. eq. 1 in manuscript
        A[i,i]=k*L       # u_a(t[i])  
        A[i,i+n_model]=-k*L  # u_m(t[i])
        
        # this is the derivative -du_m(t)/d_t,
        # we make sure this derivative operator is numerically time-symmetric everywhere it can be, and asymmetric
        # only at the edges
        if i > 0 and i < (n_model-1):
            # symmetric derivative if not at edge
            # this cancels out
            #A[i,i+n_t]+=-L*0.5/dt       # -0.5 * (u_m(t)-u_m(t-dt))/dt           
            A[i,i+n_model-1]+=L*0.5/dt 
            A[i,i+n_model+1]+=-L*0.5/dt     # -0.5 * (u_m(t+dt)-u_m(t))/dt
            # this cancels out
            #A[i,i+n_t]+=L*0.5/dt
        elif i == n_model-1:
            # at edge, the derivative is not symmetric
            A[i,i+n_model]+=-L*1.0/dt                
            A[i,i+n_model-1]+=L*1.0/dt
        elif i == 0:
            # at edge, the derivative is not symmetric
            A[i,i+n_model]+=L*1.0/dt                
            A[i,i+n_model+1]+=-L*1.0/dt
                    
    # measurements u_m(t_1) ... u_m(t_N)
    # weight based on error standard deviation
    idx=n.arange(n_model,dtype=n.int)
    for i in range(n_meas):
        if i not in missing_idx:
            # linear interpolation between model points
            dist=n.abs(t_model - t_meas[i])
            w=(dist<dt)*(1-dist/dt)/sigma[i]
            A[i+n_model,idx+n_model] = w
            m[i+n_model]=u_m[i]/sigma[i]

    # smoothness regularization using tikhonov 2nd order difference
    for i in range(n_model-2):
        A[i+n_model+n_meas,i+0] =      smoothness/dt
        A[i+n_model+n_meas,i+1] = -2.0*smoothness/dt
        A[i+n_model+n_meas,i+2] =      smoothness/dt
        m[i+n_model+n_meas]=0.0
        
    # return theory matrix
    return(A,m)

#Simulate step change in ambient concentration
def test_ua(t,t_on=1.0,u_a0=1.0,u_a1=1.0):
    """ simulated measurement """
    # this is a simulated true instantaneous concentration
    # simple "on" at t_on model and gradual decay
    u_a=n.zeros(len(t))
    u_a=n.linspace(u_a0,u_a1,num=len(t))
    # turn "on"
    # u_a[t<t_on]=0.0
    # smooth the step a little bit
    u_a=n.real(n.fft.ifft(n.fft.fft(n.repeat(1.0/5,5),len(u_a))*n.fft.fft(u_a)))
    u_a[t<t_on]=0.0    
    return(u_a)

#Create forward model (convolution)
def forward_model(t,u_a,k=1.0,u_m0=0.0):
    """ forward model """
    # evaluate the forward model, which includes slow diffusion
    # t is time
    # u_a is the concentration
    # k is the growth coefficient
    # u_m0 is the initial boundary condition for the diffused quantity
    u_m = n.zeros(len(t))
    dt = n.diff(t)[0]
    if len(k):
        for i in range(1,len(t)):
            u_m[i]=u_a[i] - (u_a[i]-u_m[i-1])*n.exp(-k[i]*dt)
    else:
        for i in range(1,len(t)):
            u_m[i]=u_a[i] - (u_a[i]-u_m[i-1])*n.exp(-k*dt)
    return(u_m)

#Simulate data collected in toy-model
def sim_meas(t,u_a,k=1.0,u_m0=0.0):
    # simulate measurements, including noise
    u_m=forward_model(t,u_a,k=k,u_m0=u_m0)
    # a simple model for measurement noise, which includes
    # noise that is always there, and noise that depends on the quantity
    noise_std = u_m*0.01 + 0.001
    m=u_m + noise_std*n.random.randn(len(u_m))
    return(m,noise_std)

#Function that finds delta t using cubic spline approx. to max curvature point
#in L-curve
def find_kink(err_norms, #  Solution norm (first-order differences of the maximum a posteriori solution)
              sol_norms, #  Model fit residual norm (sum of residuals between model measurements and real measurements)
              num_sol, # The number of timesteps in the solution
              n_models): #Number of different models to be tested
    #Make log-versions of error norm and solution norm
    err_norms_lg = n.log(err_norms)
    sol_norms_lg = n.log(sol_norms)        
    #Smoothing because the shift in time-steps between sparse model grid and sharp features in the
    #data makes the regularization jump back and forth a bit locally.
    arg_err0 = n.argsort(err_norms_lg)
    #arg_err0 = n.arange(0,len(err_norms_lg))
    #arg_err0 = n.arange(0,len(err_norms_lg))
    #Find nearest odd number for smoothing
    sw = int((n.ceil(num_sol/4)//2)*2+1) #Returns nearest odd number up   
    errlog_smooth = n.convolve(err_norms_lg[arg_err0],n.ones(sw)/sw,mode='valid')
    #Check out different counting due to smoothing. 
    sollog_smooth = n.convolve(sol_norms_lg[arg_err0],n.ones(sw)/sw,mode='valid')
    #Make regularized spline and find max point:
    splmooth = 0.1 #This regularize the spline through weighting of third derivatives
    arcs = n.arange(errlog_smooth.shape[0]) #Define arclength (same for both, does not impact where max curv is)
    std = splmooth * n.ones_like(errlog_smooth) #This regularizes the spline to
    #be smooth by weighting the third derivatives of the spline
    #Make splines of err_norms and sol_norms
    spl_err_n = unisp(arcs, errlog_smooth, k=4, w=1 / n.sqrt(std))
    spl_sol_n = unisp(arcs, sollog_smooth, k=4, w=1 / n.sqrt(std))

    #Calculate derivatives, curvatures and 2d curvature. 
    der_err_n = spl_err_n.derivative(1)(arcs)
    curv_err_n = spl_err_n.derivative(2)(arcs)
    der_sol_n = spl_sol_n.derivative(1)(arcs)
    curv_sol_n = spl_sol_n.derivative(2)(arcs)
    curvature = abs((der_err_n*curv_sol_n-der_sol_n*curv_err_n)/n.power(der_err_n**2+der_sol_n**2,1.5))

    #Find max curvature location:
    idx = n.abs(curvature-max(curvature)).argmin()
    n_model = int(n_models[[arg_err0[idx+(int(sw/2-0.5))]]])
   
    return(n_model)

#Function that estimates concentration
def estimate_concentration(u_m, #Measurements 
                           u_m_stdev, #Uncertainty of measurements  (same size as u_m)
                           t_meas, #Time vector for measurements
                           k, #Growth coefficient
                           n_model=400, #Number of model points 
                           smoothness=0, #The amount of smoothness. Set to zero to turn off Tikhonov regularization
                           L_factor=1e5,
                           calc_var=True): #True if you want to model error propagation

    # how many grid points do we have in the model
    t_model=n.linspace(n.min(t_meas),n.max(t_meas),num=n_model)
    
    A,m_v=diffusion_theory(u_m,k=k,t_meas=t_meas,t_model=t_model,sigma=u_m_stdev,smoothness=smoothness,L_factor=L_factor)
    
    #least squares solution
    #xhat=so.nnls(A,m_v)[0] #If you need only positive values. 
    xhat=n.linalg.lstsq(A,m_v)[0]        
    
    u_a_estimate=xhat[0:n_model]
    u_m_estimate=xhat[n_model:(2*n_model)]    

    if calc_var:
        # a posteriori error covariance
        Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))

        # standard deviation of estimated concentration u_a(t)
        std_p=n.sqrt(n.diag(Sigma_p))
        u_a_std=std_p[0:n_model]
        u_m_std=std_p[n_model:(2*n_model)]
    else:
        u_a_std=n.repeat(0,n_model)
        u_m_std=n.repeat(0,n_model)        
    
    return(u_a_estimate, u_m_estimate, t_model, u_a_std, u_m_std)


#Toy model simulation test function. Automatically produces an L-Curve plot
def unit_step_test(k=0.1, #growth coefficient
                   missing_meas=False, #missing measurement trigger
                   missing_t=[14,16], #missing measurement location
                   pfname="unit_step.png", #name of output figure
                   n_model = 'auto', #model complexity (number of model-points, i.e. delta T)
                   delta_ts = 'auto', #Range of delta-ts used to estimate L-curve. Specified as 'auto' (default), [min,max] of desired delta_t, or array of values
                   num_sol = 50): #Number of solutions used in L-curve
    
    t=n.linspace(0,50,num=500) #size of simulation sample
    
    if missing_meas:
        idx=n.arange(len(t))
        missing_idx=n.where( (t[idx]>missing_t[0]) & (t[idx]<missing_t[1]))[0]
    else:
        missing_idx=[]
        
    #create step-change 
    u_a=test_ua(t,t_on=5,u_a0=1.0,u_a1=1.0)
    
    # Simulate measurement affected by diffusion
    # Test effect of drift in permeation
    #k_meas = k-n.arange(0,len(t))*0.0003 #Test effect of sensor drift
    #k_meas = k+n.zeros(len(t)) #If abrupt drift... 
    #k_meas[100:len(k_meas)]=k_meas[100:len(k_meas)]-0.05
    k_meas = k*n.ones(len(t)) #If sensor works properly without drift. 
    u_m=forward_model(t,u_a,k=k_meas)
    m,noise_std=sim_meas(t,u_a,k=k_meas)
       
    #Define which delta ts to estimate in L-curve. Max and min first
    if delta_ts == 'auto':
        if len(u_m)/2 <= 4000:
            maxn_model = len(u_m)/2
        else:
            maxn_model = 2000
    #Write a function which creates the number of delta ts to create L-curve from    
    #Base this on exponential increase in number per step and the number of desired estimates 
        base_x = n.log2(maxn_model/10)/num_sol
        n_models = list() 
        for tmpidx in range(0,num_sol): n_models.append(int((2**(tmpidx*base_x)*10)))
        n_models = n.array(n_models)
        #n_models = 10*n.exp(n.linspace(0,num_sol-1,num_sol)*base_x)
        #n_models = n.linspace(minn_model,maxn_model,num_sol)
    elif len(n_models) >= 3: #If an array of values are provided.
        n_models = delta_ts
    else:
        n_models = n.linspace(delta_ts[0],delta_ts[1],num_sol)
   
    N=len(n_models) #Model grid size
    err_norms=n.zeros(N) #Define vectors for error norm...
    sol_norms=n.zeros(N) #and fit residual norm
                   
    #Make all estimations for L-curve analysis: 
    for i in range(len(n_models)):
        # don't calc a posteriori variance here, to speed things up
        u_a_est, u_m_est, t_modeln, u_a_std, u_m_std= estimate_concentration(m, noise_std, t, k, n_model=n.int(n_models[i]), smoothness=0.0,calc_var=False)
        
        um_fun=sint.interp1d(t_modeln,u_m_est) #Returns a function that interpolates
        err_norms[i]=n.sum(n.abs(um_fun(t) - m)**2.0)
        # Calculate the fit residual norm
        #sol_norms[i]=n.sum(n.abs(n.diff(n.diff(u_a_est)))**2.0) #OLD
        sol_norms[i] = n.sum(n.abs(n.diff(u_a_est)))**2.0 #New error norm
        
        print("Number of model points=%d fit residual norm %1.2f norm of solution second difference %1.2f dt=%1.2f (seconds)"%(n_models[i],err_norms[i],sol_norms[i], 24*3600*(n.max(t)-n.min(t))/float(n_models[i]) ))
    
    #If you want automatic selection of delta-t.
    if n_model == 'auto':
        n_model = find_kink(err_norms,sol_norms,num_sol,n_models)
         
    #Add time grid estimation depending on n_model (number of model points)
    t_model=n.linspace(n.min(t),n.max(t),num=n_model)
 
    # create theory matrix
    A,m_v=diffusion_theory(m,
                           t_meas=t,
                           missing_idx=missing_idx,
                           k=k,
                           t_model=t_model,
                           sigma=noise_std,
                           smoothness=0.00)

    #Linear least squares solution.
    xhat=n.linalg.lstsq(A,m_v)[0]
    
    #take out u_a and u_m estimates
    u_a_estimate=xhat[0:n_model]
    u_m_estimate=xhat[n_model:(2*n_model)]
    
    #...uncertainty estimate
    Sigma_p=n.linalg.inv(n.dot(n.transpose(A),A))
    u_a_std=n.sqrt(n.diag(Sigma_p)[0:n_model])
    
    um_fun=sint.interp1d(t_model,u_m_estimate) #Returns a function that interpolates
    sol_err_norm=n.sum(n.abs(um_fun(t) - m)**2.0)
    # Calculate the fit residual norm
    #sol_norms[i]=n.sum(n.abs(n.diff(n.diff(u_a_est)))**2.0) #OLD
    sol_sol_norm = n.sum(n.abs(n.diff(u_a_estimate)))**2.0 #New fit residual norm
    
    #Plot data
    plt.figure()
    plt.plot(t,u_a,label="True $u_a(t)$",color="orange")
    plt.plot(t,u_m,label="True $u_m(t)$",color="brown")    
    plt.plot(t_model,u_a_estimate,color="blue",label="Estimate $\\hat{u}_a(t)$ @ \Delta t =")
    plt.plot(t_model,u_a_estimate+2.0*u_a_std,color="lightblue",label="2-$\\sigma$ uncertainty")
    lower_bound=u_a_estimate-2.0*u_a_std
    lower_bound[lower_bound<0]=0.0
    plt.plot(t_model,lower_bound,color="lightblue")
    plt.plot(t_model,u_m_estimate,label="Estimate $\\hat{u}_m(t)$",color="purple")
    idx=n.arange(len(t),dtype=n.int)
    idx=n.setdiff1d(idx,missing_idx)
    plt.plot(t[0::4],m[0::4],".",label="Missing measurement at t = 15-17",color="red")
    
    plt.xlabel("Time")
    plt.ylabel("Property")
    plt.legend(ncol=2)
    plt.legend('')
    plt.ylim([-0.2,2.0])
    plt.tight_layout()
    plt.savefig(pfname)
    plt.show()
    
    #Plot L-curve
    plt.figure()
    plt.title("L-curve")
    plt.loglog(err_norms,sol_norms,"*")
    plt.loglog(sol_err_norm,sol_sol_norm,"*",color="Red")
    # plot how many samples are used to represent the concentration
    plt.text((sol_err_norm),(sol_sol_norm),"$\Delta t$=%.2f"%(n.abs(((max(t)-min(t))/n_model))))
    #plt.text((err_norms[idxs[i]]),(sol_norms[idxs[i]]),"$\Delta t$=%.2f"%(n.abs(((max(t)-min(t))/n_models[idxs[i]]))))    
    #plt.text((err_norms[idxs[i]]),(sol_norms[idxs[i]]),"$N$=%d"%(n_models[idxs[i]]))
    plt.xlabel("Fit error residual $E_m$")#,FontSize = 20)
    plt.ylabel("Norm of solution $E_s$")#, FontSize = 20)    
    plt.xlim((-4,-3))
    plt.show()
    
    #Look at residuals    
    N=n_model
    u_a_est, u_m_est, t_modeln, u_a_std, u_m_std= estimate_concentration(m, noise_std, t, k, n_model=N, smoothness=0.0,calc_var=False)

    f = plt.figure()
    um_fun=sint.interp1d(t_modeln,u_m_est)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    plt.plot(t,(um_fun(t) - m))
    plt.ylabel("Fit residual m-Vu")
    plt.xlabel("Time")
    plt.xlim((0,50))
    plt.ylim((-0.05,0.06))
    plt.show()
        
    print("Delta t equals")
    print(n.abs(((max(t)-min(t))/n_model)))

def field_example_tikhonov(): #same as sensor_example just with tikhonov regularization
    # read lab data
    d=sio.loadmat("fielddata.mat") 
    t=n.copy(d["time"])[:,0]
    u_slow=n.copy(d["slow"])[:,0]
    u_fast=n.copy(d["fast"])[:,0]

    # remove nan values
    idx=n.where(n.isnan(u_slow)!=True)[0]
    # use all measurements. make this smaller if you want to speed up things
    n_meas=len(idx)
    # use this many points to model the concentration.
    n_model=1500
    
    m_u_slow=u_slow[idx[0:n_meas]]
    m_t=t[idx[0:n_meas]]
    
    k=(60.0*24.0)/30.0

    # estimate measurement error standard deviation from differences
    sigma_est = n.sqrt(n.var(n.diff(m_u_slow)))
    sigma=n.repeat(sigma_est,len(m_t))
    
    # L-curve
    # try out different regularization parameters
    # and record the norm of the solution's second derivative, as
    # well as the norm of the fit error residual
    sms = 10**(n.linspace(-6,-1,num=10))
    n_L=len(sms)
    err_norms=n.zeros(n_L)
    sol_norms=n.zeros(n_L)    
    for i in range(n_L):
        # don't calc a posteriori variance here, to speed things up
        u_a_est, u_m_est, t_model, u_a_std, u_m_std= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n_model, smoothness=sms[i],calc_var=False)

        um_fun=sint.interp1d(t_model,u_m_est)
        err_norms[i]=n.sum(n.abs(um_fun(m_t) - m_u_slow)**2.0)
        # norm of the second order Tikhonov regularization (second derivative)
        sol_norms[i]=n.sum(n.abs(n.diff(n.diff(u_a_est)))**2.0)
        print("Trying smoothness 10**%1.2f fit residual norm %1.2f norm of solution second difference %1.2f"%(n.log10(sms[i]),err_norms[i],sol_norms[i]))

    plt.title("L-curve")
    plt.loglog(err_norms,sol_norms,"*")
    for i in range(n_L):
        plt.text(err_norms[i],sol_norms[i],"s=%1.1f"%(n.log10(sms[i])))
    plt.xlabel("Fit error residual $||m-Ax||^2$")
    plt.ylabel("Norm of solution $||Lx||^2$")    
    plt.show()

    sm=10**(-4.0)
    print("fitting")    
    u_a_est, u_m_est, t_model, u_a_std, u_m_std= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n_model, smoothness=sm)    

    print("plotting")
    plt.plot(m_t,m_u_slow)
    plt.plot(t,u_fast)
    plt.plot(t_model,u_a_est,color="green")
    plt.plot(t_model,u_a_est+2*u_a_std,color="lightgreen")
    plt.plot(t_model,u_a_est-2*u_a_std,color="lightgreen")        
    plt.show()

def field_example_model_complexity():
    # Use only model sparsity (sampling rate of the model) as the a priori assumption
    # We set smoothness to zero, which turns off Tikhonov regularization.
    # read data
    d=sio.loadmat("time.mat") #data matrix
    #t=n.copy(d["time"])[:,0]
    #u_slow=n.copy(d["slowsens"])[:,0]
    #u_fast=n.copy(d["fastsens"])[:,0]
    
    #For time.mat
    t=n.copy(d["time"])[:] #time vector
    u_slow=n.copy(d["slowsens"])[:] #slow sensor data
    u_fast=n.copy(d["fastsens"])[:] #fast sensor data
        
    k=(60.0*24.0)/40.0 #Growth coefficient
    n_model = 'auto' #model complexity (number of model-points, i.e. delta T)
    delta_ts = 'auto' #Range of delta-ts used to estimate L-curve. Specified as [min,max] of desired delta_t
    num_sol = 30
        
    # remove nan values
    idx=n.where(n.isnan(u_slow)!=True)[0]
    # use all measurements. make this smaller if you want to speed up things
    n_meas=len(idx)
    # use this many points to model the concentration.
    
    m_u_slow=u_slow[idx[0:n_meas]]
    m_t=t[idx[0:n_meas]]
    
    # estimate measurement error standard deviation from differences
    sigma_noise_floor = n.sqrt(n.var(n.diff(m_u_slow)))#+0.03*(m_u_slow[1:len(m_u_slow)])))
    
    
    sigma = 0.015*(m_u_slow[0:len(m_u_slow)]) #Sensor accuracy
    #for ind in range(len(sigma)):
     #   if sigma[ind] < sigma_noise_floor:
      #      sigma[ind] = sigma_noise_floor
    
    # L-curve
    # try out different regularization parameters
    # and record the norm of the solution's second derivative, as
    # well as the norm of the fit error residual
    
    #Define which delta ts to estimate in L-curve. Max and min first
    if delta_ts == 'auto':
        if len(m_u_slow)/2 <= 4000:
            maxn_model = len(m_u_slow)/2
        else:
            maxn_model = 2000
    #Function which creates the number of delta ts to create L-curve from    
    #Base this on exponential increase in number per step and the number of desired estimates 
        base_x = n.log2(maxn_model/10)/num_sol
        n_models = list() 
        for tmpidx in range(0,num_sol): n_models.append(int((2**(tmpidx*base_x)*10)))
        n_models = n.array(n_models)
    elif len(n_models) >= 3: #If an array of values are provided.
        n_models = delta_ts
    else:
        n_models = n.linspace(delta_ts[0],delta_ts[1],num_sol)
   
    N=len(n_models) #Model grid size
    err_norms=n.zeros(N) #Define vectors for error norm...
    sol_norms=n.zeros(N) #and fit residual norm
                   
    #Make all estimations for L-curve analysis: 
    for i in range(len(n_models)):
        # don't calc a posteriori variance here, to speed things up
        u_a_est, u_m_est, t_modeln, u_a_std, u_m_std= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n.int(n_models[i]), smoothness=0.0,calc_var=False)
        
        um_fun=sint.interp1d(t_modeln,u_m_est) #Returns a function that interpolates
        err_norms[i]=n.sum(n.abs(um_fun(m_t) - m_u_slow)**2.0) #Model fit residual norm
        # Calculate the fit residual norm
        #sol_norms[i]=n.sum(n.abs(n.diff(n.diff(u_a_est)))**2.0) #OLD
        sol_norms[i] = n.sum(n.abs(n.diff(u_a_est)))**2.0 #New solution norm
        
        print("Number of model points=%d fit residual norm %1.2f norm of solution second difference %1.2f dt=%1.2f (seconds)"%(n_models[i],err_norms[i],sol_norms[i], 24*3600*(n.max(t)-n.min(t))/float(n_models[i]) ))
                
    #Search for delta_t
    if n_model == 'auto':
        n_model = find_kink(err_norms,sol_norms,num_sol,n_models)
       
    #Make a final calculation using the delta t found or defined
    N=n_model
    u_a_est, u_m_est, t_model, u_a_std, u_m_std= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=N, smoothness=0.0)
   
    #Get solution and error norms for the chosen delta t
    um_fun=sint.interp1d(t_model,u_m_est) #Returns a function that interpolates
    sol_err_norm = n.sum(n.abs(um_fun(m_t) - m_u_slow)**2.0)
    sol_sol_norm = n.sum(n.abs(n.diff(u_a_est)))**2.0

    #Make L-curve plot
    plt.figure()
    plt.title("L-curve")
    plt.loglog(err_norms,sol_norms,"*")
    plt.loglog(sol_err_norm,sol_sol_norm,"*",color="Red")
       
    # plot how many samples are used to represent the concentration
    plt.text((sol_err_norm),(sol_sol_norm),"$\Delta t$=%.2f"%(60*60*24*n.abs(((max(t_model)-min(t_model))/n_model))))
    # FontSize doesn't work on linux for some reason?
    plt.xlabel("Fit error residual $E_m$")#,FontSize = 20)
    plt.ylabel("Norm of solution $E_s$")#, FontSize = 20)    
    plt.show()
    
    #Make plot of result estimate
    print("plotting")
    plt.figure()
    plt.plot(m_t,m_u_slow,".")
    plt.plot(t,u_fast)

    dt=(n.max(t_model)-n.min(t_model))/float(N)
    
    plt.title("Number of model points N=%d $\Delta t=%1.2f (s)$"%(N,dt*24*3600.0))
    plt.plot(t_model,u_a_est,color="green")
    plt.plot(t_model,u_a_est+2*u_a_std,color="lightgreen")
    plt.plot(t_model,u_a_est-2*u_a_std,color="lightgreen")        
    plt.show()
        
def deconv_master(u_slow,t,k,
                  sigma='auto',
                  delta_t='auto',
                  delta_range = 'auto',
                  num_sol=30,
                  L_factor=1e5,
                  N = 'auto'):
    ''' Function that deconvolves sensor data and returns vectors containing
    the estimated corrected data (u_a_estimate), measurement estimate (u_m_estimate), 
    model time (model_time), uncertainty estimate (standard deviation) 
    std_uncertainty_estimate), model fit residuals (fit_resids)).
    Function also provides an L-curve plot, fit residual plot and estimate plot.
    
    u_a_estimate,u_m_estimate,model_time,std_uncertainty_estimate,fit_resids=
    deconv_master(data,time,k,delta_t='auto',delta_range='auto',num_sol=30)
       
    Parameters: 
        
        u_slow: (N,)array_like 
            Convoluted sensor data
        
        t: (N,)array_like 
            time vector for convoluted sensor data in seconds
        
        k: (N,)array_like or float
            1/tau63, where tau63 is the response time. Growth coefficient.
       
        sigma: (N,)array_like, float or None
            Measurement uncertainty. Given as array species uncertainty of each
            measurement. Float gives the same measurement uncertainty to all points
            None makes the algorithm estimate noise in the measurements using finite
            difference. 
       
        delta_t: float or None
            Sets model complexity. Single float setting the desired resolution
        of the modelled concentration in seconds. Default is 'auto', which finds the 
        optimal resolution using L-curve analysis
        
        delta_range: list or None
            Defines the range of delta ts you want to use in L-curve plot and
        automated delta_t selection if delta_t='auto'. Default is 'auto' where 
        delta t_range is set from 10 model points to len(data)/2 up to 2000 and 
        the number of delta_t to check as defined by num_sol. 
        
        num_sol: integer or None
            Sets the number of solutions to estimate to make the L-curve. default 
            is 30.
            
        N: Integer
            Sets the number of model points. default is "auto", which uses the 
            delta_t input overrides any delta_t if specified.
        
    '''
     
    ### remove nan values
    idx=n.where(n.isnan(u_slow)!=True)[0]
    n_meas=len(idx)
    # use this many points to model the concentration.
    m_u_slow=u_slow[idx[0:n_meas]]
    m_t=t[idx[0:n_meas]]
    
    ### Get measurement error standard deviation from differences if uncertainty is not
    #specified: 
    if sigma == 'auto':
        sigma = n.sqrt(n.var(n.diff(m_u_slow)))
    elif len(sigma)==1:
        sigma = sigma*n.ones(len(m_u_slow))
    elif len(sigma) == len(m_u_slow):
        sigma = sigma
        
    ###
    # L-curve analysis and if auto is on, find best delta_t 
    # In other words: try out different regularization parameters
    # and record the norm of the solution's second derivative, as
    # well as the norm of the fit error residual
    
    #Define which delta ts to estimate for in L-curve... If this is not 
    #Function which creates the number of delta ts to create L-curve from    
    #Base this on exponential increase in number per step and the number of desired estimates 
    #bacause - high resolution estimates takes a lot more time and from tests
    #the solution norm gets bad really fast when the resolution starts to get too coarse, so no need
    #for that many points: 
    if delta_range == 'auto':
        if len(m_u_slow)/2 <= 4000:
            maxn_model = len(m_u_slow)/2
        else:
            maxn_model = 2000
        #model for making list of delta_ts that are to be used for estimates:
        base_x = n.log2(maxn_model/10)/num_sol
        n_models = list() 
        for tmpidx in range(0,num_sol+1): n_models.append(int((2**((tmpidx)*base_x)*10)))
        n_models = n.array(n_models)
    else: #If an array of values are provided.
        tmp = max(t)-min(t)
        n_models = [tmp/x for x in delta_range] #Convert from delta_t to n_models
        
    err_norms=n.zeros(len(n_models)) #Define vectors for error norm...
    sol_norms=n.zeros(len(n_models)) #and fit residual norm

    n_models=[]
    #Make all estimations for L-curve analysis: 
    for i in range(len(n_models)):
        # don't calc a posteriori variance here, to speed things up
        u_a_est, u_m_est, t_modeln, u_a_std, u_m_std= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n.int(n_models[i]), smoothness=0.0,calc_var=False,L_factor=L_factor)

        
        um_fun=sint.interp1d(t_modeln,u_m_est) #Returns a function that interpolates
        err_norms[i]=n.sum(n.abs(um_fun(m_t) - m_u_slow)**2.0)
        # Calculate the fit residual norm
        #sol_norms[i]=n.sum(n.abs(n.diff(n.diff(u_a_est)))**2.0) #OLD
        sol_norms[i] = n.sum(n.abs(n.diff(u_a_est)))**2.0 #New fit residual norm
        
        print("Number of model points=%d fit residual norm %1.2f norm of solution second difference %1.2f dt=%1.2f (seconds)"%(n_models[i],err_norms[i],sol_norms[i], 24*3600*(n.max(t)-n.min(t))/float(n_models[i]) ))
                
    
    if N=='auto':
        if delta_t == 'auto': #Run find_kink to detect optimal delta t if delta_t is not specified. 
            n_model = find_kink(err_norms,sol_norms,num_sol,n_models)    
        else: #If delta_t is specified 
            n_model = n.int((max(t)-min(t))/delta_t) #If delta:t is specified
    else:
        n_model = N #If N is specified
        
    ### Make final estimate using the delta_t found or defined:
    u_a_est, u_m_est, t_model, u_a_std1, u_m_std1= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n_model, smoothness=0.0, L_factor=1e5)
    u_a_est, u_m_est, t_model, u_a_std2, u_m_std2= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n_model, smoothness=0.0, L_factor=1e4)
    u_a_est, u_m_est, t_model, u_a_std3, u_m_std3= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n_model, smoothness=0.0, L_factor=1e3)
    u_a_est, u_m_est, t_model, u_a_std4, u_m_std4= estimate_concentration(m_u_slow, sigma, m_t, k, n_model=n_model, smoothness=0.0, L_factor=1e1)
    plt.plot(t_model,u_a_std1/2,".",label="$\gamma=10^5$")
    plt.plot(t_model,u_a_std2/2,".",label="$\gamma=10^4$")
    plt.plot(t_model,u_a_std3/2,".",label="$\gamma=10^3$")
    plt.plot(t_model,u_a_std4/2,".",label="$\gamma=1$")
    plt.legend()
    plt.show()
    
    
    ### Get error and solution norms for the chosen delta t
    um_fun=sint.interp1d(t_model,u_m_est) #Returns a function that interpolates
    sol_err_norm = n.sum(n.abs(um_fun(m_t) - m_u_slow)**2.0)
    sol_sol_norm = n.sum(n.abs(n.diff(u_a_est)))**2.0

    ### Plotting
    #Make L-curve plot
    plt.figure()
    plt.title("L-curve")
    plt.loglog(err_norms,sol_norms,"*")
    plt.loglog(sol_err_norm,sol_sol_norm,"*",color="Red")
    # plot where the used delta_t is in the L-curve: 
    plt.text((sol_err_norm),(sol_sol_norm),"$\Delta t$=%.2f"%(n.abs(((max(t_model)-min(t_model))/n_model))))
    plt.xlabel("Fit error residual $E_m$")#,FontSize = 20)
    plt.ylabel("Norm of solution $E_s$")#, FontSize = 20)    
    plt.show()
    
    # plot fit residuals
    um_fun=sint.interp1d(t_model,u_m_est)
    plt.figure()
    resid=um_fun(m_t) - m_u_slow
    #plt.plot(m_t[good_m_idx],um_fun(m_t[good_m_idx]) - m_u_slow[good_m_idx],"o")
    plt.plot(m_t,um_fun(m_t)-m_u_slow,"x")
    plt.plot(m_t,um_fun(m_t)-m_u_slow)
    #This is only for the limits in the resid-plot... 
    std_est=n.median(n.abs(um_fun(m_t)-m_u_slow))
    plt.axhline(3*std_est)
    plt.axhline(-3*std_est)
    plt.xlabel('Fit error residual')
    plt.title('Fit residuals')
    
    #Make plot of resulted estimate
    plt.figure()
    plt.plot(m_t,m_u_slow,".",label="Convoluted")
    plt.plot(t_model,u_a_est,color="green",label="De-convoluted")
    plt.plot(t_model,u_a_est+2*u_a_std,color="lightgreen",label="Standard dev.")
    plt.plot(t_model,u_a_est-2*u_a_std,color="lightgreen")
    plt.ylabel('Quantity')        
    plt.xlabel('Seconds')
    plt.legend()
    dt=(n.max(t_model)-n.min(t_model))/float(n_model)    
    plt.title("Number of model points N=%d $\Delta t=%1.2f (s)$"%(n_model,dt))   
    plt.show()
    
    return(u_a_est,u_m_est,t_model,u_a_std,resid)
          
if __name__ == "__main__":
    #unit_step_test(pfname="unit_step.png",num_sol=100)    
    #unit_step_test(missing_meas=True,pfname="unit_step_missing.png")    
    #use model sparsity (finite number of samples) to regularize the solution   
    #field_example_model_complexity()
    #Use deconv_master to deconvolve data from the field experiment:
        
    
    d=sio.loadmat("fielddata.mat")
    t=n.copy(d["time"])[:,0]
    u_slow=n.copy(d["slow"])[:,0]
    u_fast=n.copy(d["fast"])[:,0]
    t = t-min(t)
    t = t*86400
    t_fast = t
    sigma = 0.03*u_slow
    k = 1/(39*60) #In seconds because time vector is in seconds
    
   # deconv_master(u_slow,t,k,sigma = sigma,delta_t=500)#, N = 120,num_sol=1)
   
    u_slow = u_slow[0:len(u_slow):1]
    t = t[0:len(t):1]   
    sigma = sigma[0:len(sigma):1]
    
    u_a_est,u_m_est,t_model,u_a_std,resid = deconv_master(u_slow,t,k,delta_t=55,num_sol=1,sigma = sigma)#, N = 120,num_sol=1)

 
   
