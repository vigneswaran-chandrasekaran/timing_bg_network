import torch, os
import numpy as np
from classifier_model import BG
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Define logistic function
def logistic_function(x, L, k, x0, b):
    """
    Logistic function: L / (1 + exp(-k*(x-x0))) + b
    L: maximum value of the curve
    k: steepness of the curve
    x0: x-value of the sigmoid's midpoint
    b: y-intercept
    """
    return L / (1 + np.exp(-k * (x - x0))) + b

# n_iter = 100

# dt = 0.002
# num_steps = 1000
# rl_scaler = 1
# units = 64

# min_omega = 30
# max_omega = 60

# input_dim = 1
# action_dim = 2

# PD_FLAG = 0
# load_folder = f'tbi_ms_pd{PD_FLAG}/'

# if not os.path.exists(load_folder):
#     raise FileNotFoundError(f"Load folder {load_folder} does not exist.")

# jumbo_mon = []
# for iter in range(n_iter):

#     iter_mon = []    

#     for d in np.arange(0.4, 1.8, 0.2):      
#         offset = int(d/(dt * rl_scaler))
#         t_stimuli = np.arange(0, offset*dt*rl_scaler, dt)
#         t_noise = np.arange(0, (num_steps-offset)*dt*rl_scaler, dt)
#         state_stimuli = 5*np.sin(2*np.pi*10*t_stimuli)
#         state_noise = np.sin(2*np.pi*10*t_noise) 
#         state = np.hstack((state_stimuli, state_noise))
#         state = torch.tensor(state, dtype=torch.float32).reshape(-1, 1, 1)
 
#         model = BG(input_dim, action_dim, units, dt,
#                    min_omega, max_omega, pd_flag=PD_FLAG)
#         model.load_state_dict(torch.load(f'{load_folder}model.pth'))
#         model.d1.omegas.data = torch.tensor(np.load(f'{load_folder}d1_omegas.npy'))
#         model.d2.omegas.data = torch.tensor(np.load(f'{load_folder}d2_omegas.npy'))
#         (pred, value, zipper) = model(state)
#         pred_class = torch.argmax(pred, 1).item()
#         iter_mon.append(pred_class)

#     jumbo_mon.append(iter_mon)
# jumbo_mon = np.array(jumbo_mon)

# print("Completed Normal Control")
# PD_FLAG = 1
# load_folder = f'tbi_ms_pd{PD_FLAG}/'
# if not os.path.exists(load_folder):
#     raise FileNotFoundError(f"Load folder {load_folder} does not exist.")

# pd_jumbo_mon = []
# for iter in range(n_iter):

#     iter_mon = []    

#     for d in np.arange(0.4, 1.8, 0.2):        
#         offset = int(d/(dt * rl_scaler))
#         t_stimuli = np.arange(0, offset*dt*rl_scaler, dt)
#         t_noise = np.arange(0, (num_steps-offset)*dt*rl_scaler, dt)
#         state_stimuli = 5*np.sin(2*np.pi*10*t_stimuli)
#         state_noise = np.sin(2*np.pi*10*t_noise) 
#         state = np.hstack((state_stimuli, state_noise))
#         state = torch.tensor(state, dtype=torch.float32).reshape(-1, 1, 1)
  
#         pd_model = BG(input_dim, action_dim, units, dt,
#                    min_omega, max_omega, pd_flag=PD_FLAG)
#         pd_model.load_state_dict(torch.load(f'{load_folder}model.pth'))
#         pd_model.d1.omegas.data = torch.tensor(np.load(f'{load_folder}d1_omegas.npy'))
#         pd_model.d2.omegas.data = torch.tensor(np.load(f'{load_folder}d2_omegas.npy'))
#         (pred, value, zipper) = pd_model(state)
#         pred_class = torch.argmax(pred, 1).item()
#         iter_mon.append(pred_class)

#     pd_jumbo_mon.append(iter_mon)
# pd_jumbo_mon = np.array(pd_jumbo_mon)

# ######################### sch

# load_folder = f'tbi_ms_sc{0}/'
# if not os.path.exists(load_folder):
#     raise FileNotFoundError(f"Load folder {load_folder} does not exist.")

# sc_jumbo_mon = []
# for iter in range(n_iter):

#     iter_mon = []    

#     for d in np.arange(0.4, 1.8, 0.2):        
#         offset = int(d/(dt * rl_scaler))
#         t_stimuli = np.arange(0, offset*dt*rl_scaler, dt)
#         t_noise = np.arange(0, (num_steps-offset)*dt*rl_scaler, dt)
#         state_stimuli = 5*np.sin(2*np.pi*10*t_stimuli)
#         state_noise = np.sin(2*np.pi*10*t_noise) 
#         state = np.hstack((state_stimuli, state_noise))
#         state = torch.tensor(state, dtype=torch.float32).reshape(-1, 1, 1)
  
#         sc_model = BG(input_dim, action_dim, units, dt,
#                    min_omega, max_omega, sc_flag=1)
#         sc_model.load_state_dict(torch.load(f'{load_folder}model.pth'))
#         sc_model.d1.omegas.data = torch.tensor(np.load(f'{load_folder}d1_omegas.npy'))
#         sc_model.d2.omegas.data = torch.tensor(np.load(f'{load_folder}d2_omegas.npy'))
#         (pred, value, zipper) = sc_model(state)
#         pred_class = torch.argmax(pred, 1).item()
#         iter_mon.append(pred_class)

#     sc_jumbo_mon.append(iter_mon)
# sc_jumbo_mon = np.array(sc_jumbo_mon)

# # Prepare data for plotting
# x_data = np.arange(0.4, 1.8, 0.2)
# control_mean = np.mean(jumbo_mon, 0)
# pd_mean = np.mean(pd_jumbo_mon, 0)
# sc_mean = np.mean(sc_jumbo_mon, 0)

x_data = np.arange(0.4, 1.8, 0.2)

control_mean = np.load('control_mean.npy')
pd_mean = np.load('pd_mean.npy')
sc_mean = np.load('sc_mean.npy')

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.dpi'] = 400
# Plot original data points
plt.plot(x_data, control_mean, '-o', label='Control', alpha=1, color="green")
plt.plot(x_data, pd_mean, '-o', label='PD', alpha=1, color="orange")
plt.plot(x_data, sc_mean, '-o', label='SCZ', alpha=1, color="blue")

# Fit and plot logistic functions
try:
    # Fit logistic function to control data
    # Initial parameter guess: [L, k, x0, b]
    control_popt, _ = curve_fit(logistic_function, x_data, control_mean, 
                               p0=[1, 5, 1.1, 0], maxfev=5000)
    
    # Fit logistic function to PD data
    pd_popt, _ = curve_fit(logistic_function, x_data, pd_mean, 
                          p0=[1, 5, 1.1, 0], maxfev=5000)
    
    sc_popt, _ = curve_fit(logistic_function, x_data, sc_mean, 
                          p0=[1, 5, 1.1, 0], maxfev=5000)
    
    # Create smooth curves for plotting
    x_smooth = np.linspace(0.4, 1.65, 100)
    control_fit = logistic_function(x_smooth, *control_popt)
    pd_fit = logistic_function(x_smooth, *pd_popt)
    sc_fit = logistic_function(x_smooth, *sc_popt)
    # Plot fitted curves
    plt.plot(x_smooth, control_fit, '--', color='green', alpha=0.5, 
             linewidth=1)
    plt.plot(x_smooth, pd_fit, '--', color='orange', alpha=0.5, 
            linewidth=1)
    plt.plot(x_smooth, sc_fit, '--', color='blue', alpha=0.5, 
             linewidth=1)
    np.save('control_fit.npy', control_fit)
    np.save('pd_fit.npy', pd_fit)
    np.save('sc_fit.npy', sc_fit)
    np.save('control_mean.npy', control_mean)
    np.save('pd_mean.npy', pd_mean)
    np.save('sc_mean.npy', sc_mean)
    # Find x-values where y=0.5 for both curves
    # Interpolate to find more precise x-values
    control_x_at_05 = np.interp(0.5, control_fit, x_smooth)
    pd_x_at_05 = np.interp(0.5, pd_fit, x_smooth)
    sc_x_at_05 = np.interp(0.5, sc_fit, x_smooth)

    # Add vertical lines at y=0.5
    plt.axvline(x=control_x_at_05, color='green', linestyle=':', alpha=0.2, 
                label=f'NC BP at t={control_x_at_05:.3f}s')
    plt.axvline(x=pd_x_at_05, color='orange', linestyle=':', alpha=0.2, 
                label=f'PD BP at t={pd_x_at_05:.3f}s')
    plt.axvline(x=sc_x_at_05, color='blue', linestyle=':', alpha=0.2, 
                label=f'SCZ BP at t={sc_x_at_05:.3f}s')
    
    print(f"Control logistic parameters: L={control_popt[0]:.3f}, k={control_popt[1]:.3f}, x0={control_popt[2]:.3f}, b={control_popt[3]:.3f}")
    print(f"PD logistic parameters: L={pd_popt[0]:.3f}, k={pd_popt[1]:.3f}, x0={pd_popt[2]:.3f}, b={pd_popt[3]:.3f}")
    print(f"NC BP at t={control_x_at_05:.3f}s")
    print(f"PD BP at t={pd_x_at_05:.3f}s")
    print(f"SCZ BP at t={sc_x_at_05:.3f}s")

except Exception as e:
    print(f"Warning: Could not fit logistic functions: {e}")

plt.xlabel('Duration (s)')
plt.ylabel('Prediction (Long)')
plt.title('Temporal Bisection Task: NC, PD and SCZ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('temp_bisection_comparison.png')
