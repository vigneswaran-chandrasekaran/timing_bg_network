import torch
import numpy as np
from donn import *

class BG(nn.Module):
    
    def __init__(self, input_dim, action_dim, units, 
                 dt, min_omega, max_omega,
                 pd_flag=0, sc_flag=0):

        super(BG, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.units = units

        self.l1 = ComplexLinear(input_dim, units)

        self.d1 = SingledtResHopf(units, min_omega, max_omega, dt)
        self.d2 = SingledtResHopf(units, min_omega, max_omega, dt)

        self.dp = ComplexLinear(units, action_dim)
        self.ip = ComplexLinear(units, action_dim)

        self.critic = ComplexLinear(units, 1)
        self.pd_flag = pd_flag
        self.sc_flag = sc_flag

    def forward(self, state):
        
        t = 0
        gpi_tau = 0.01
        noise_tau = 1

        gpi = torch.zeros(1, self.action_dim)    
        
        dp_mon = []
        ip_mon = []
        d1_mon = []
        d2_mon = []
        value_mon = []
        gpi_mon = []
        delta_V_mon = []
        
        p_value = torch.tensor(0.0)
        r1 = torch.ones(1, self.units)
        phi1 = torch.rand(1, self.units) / 10
        r2 = torch.ones(1, self.units)
        phi2 = torch.rand(1, self.units) / 10

        while t < state.shape[0]:
            
            x = cReLU(self.l1(torch.complex(state[t], state[t])))

            d1out, r1, phi1 = self.d1(x, r1, phi1)
            d2out, r2, phi2 = self.d2(x, r2, phi2)

            value = torch.abs(self.critic(d1out))
            value_mon.append(value)
            delta_V = value - p_value
            
            if self.pd_flag:
                delta_V = 0.2 * delta_V
            if self.sc_flag:
                delta_V = 1.8 * delta_V

            lD1 = torch.sigmoid(5*delta_V)
            lD2 = torch.sigmoid(-5*delta_V)

            d1_mon.append(d1out.clone().detach().numpy())
            d2_mon.append(d2out.clone().detach().numpy())
            
            dp_out = cReLU(self.dp(d1out * lD1))
            ip_out = cReLU(self.ip(d2out * lD2))

            noise = noise_tau * (torch.rand(self.action_dim) * 2*lD2 - lD2)
            gpi = gpi + gpi_tau * torch.abs(dp_out - (ip_out + noise))
            
            dp_mon.append(dp_out.clone().detach().numpy())
            ip_mon.append(ip_out.clone().detach().numpy())
            gpi_mon.append(gpi.clone().detach().numpy())
            
            t = t + 1
            p_value = value
            delta_V_mon.append(delta_V.clone().detach().item()) 
        
        # value_max = torch.max(torch.stack(value_mon, 0))
        value_mon = torch.stack(value_mon, 0).clone().detach().numpy()
        
        zipper = {'d1': np.stack(d1_mon, 0), 
                  'd2': np.stack(d2_mon, 0), 
                  'dp': np.stack(dp_mon, 0), 
                  'ip': np.stack(ip_mon, 0),
                  'gpi': np.stack(gpi_mon, 0),
                  'delta_V': np.stack(delta_V_mon, 0),
                  'value': value_mon}
        
        return(torch.softmax(gpi, 1), value, zipper)
