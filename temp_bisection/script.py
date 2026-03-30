import torch, os
import numpy as np
from classifier_model import BG
from matplotlib import pyplot as plt

dt = 0.002
num_steps = 1000
rl_scaler = 1
units = 64

min_omega = 30
max_omega = 60

input_dim = 1
action_dim = 2

PD_FLAG = 0
save_folder = f'tbi_ms_sc{PD_FLAG}/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    print(f"Warning: path {save_folder} already exists! Rewriting the contents inside it")

print("NOTE: You're currently running model in PD={%d} mode" % PD_FLAG)

print(f"Duration of a trial {dt* num_steps * rl_scaler} seconds")

model = BG(input_dim, action_dim, units, dt,
           min_omega, max_omega, pd_flag=PD_FLAG, sc_flag=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

eps = np.finfo(np.float32).eps.item()

num_epochs = 200

class_seed = np.random.randint(0, 2, num_epochs)
plt.hist(class_seed)
plt.title('Class distribution')
plt.show()

running_twenty_avg = []

for episode in range(num_epochs):

    if class_seed[episode] == 0:
        desired = 0 # short        
        offset = int(0.4/(dt * rl_scaler))
        y_label = torch.tensor([1.0, 0.0]).reshape(1, -1)
    else:
        desired = 1 # long
        offset = int(1.6/(dt * rl_scaler))
        y_label = torch.tensor([0.0, 1.0]).reshape(1, -1)

    t_stimuli = np.arange(0, offset*dt*rl_scaler, dt)
    t_noise = np.arange(0, (num_steps-offset)*dt*rl_scaler, dt)
    state_stimuli = 5*np.sin(2*np.pi*10*t_stimuli)
    state_noise = np.sin(2*np.pi*10*t_noise) 
    state = np.hstack((state_stimuli, state_noise))

    state = torch.tensor(state, dtype=torch.float32).reshape(-1, 1, 1)
    (pred, value, zipper) = model(state)
    pred_class = torch.argmax(pred, 1).item()
    p = torch.max(pred, 1)[0]

    reward = 0.0 if pred_class != desired else 1.0

    cL = torch.nn.MSELoss()(torch.tensor(reward), value)
    aL = torch.nn.MSELoss()(torch.tensor(reward), p)
    # aL = -torch.log(p) * (reward - value)
    loss = cL + aL

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if len(running_twenty_avg)<20:
        running_twenty_avg.append(reward)
    else:
        running_twenty_avg.append(reward)
        running_twenty_avg.pop(0)

    if (episode+1) % 20 == 0:
        print(f'E {episode+1}, L: {loss.item()}, d: {desired}, p: {pred_class}, Avg: {np.mean(running_twenty_avg)}')
    
    if (episode+1) % 100 == 0:
        torch.save(model.state_dict(), f'{save_folder}model.pth')
        np.save(f'{save_folder}d1_omegas.npy', model.d1.omegas.clone().detach().numpy())
        np.save(f'{save_folder}d2_omegas.npy', model.d2.omegas.clone().detach().numpy())
