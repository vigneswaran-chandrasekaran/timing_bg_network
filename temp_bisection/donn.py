import torch, math
from torch.nn import functional as F, init
from torch import nn
import math
from typing import List, Tuple

def cReLU(x):
    return torch.complex(torch.relu(torch.real(x)),
                         torch.relu(torch.imag(x)))

def cLeakyReLU(x):
    return torch.complex(torch.nn.LeakyReLU()(torch.real(x)),
                         torch.nn.LeakyReLU()(torch.imag(x)))

class ComplexLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class TemporalComplexLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TemporalComplexLinear, self).__init__()
        self.linear = ComplexLinear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timesteps, _ = x.size()
        output = torch.jit.annotate(List[torch.Tensor], [])

        for t in range(timesteps):
            timestep_output = self.linear(x[t])
            output.append(timestep_output)

        return torch.stack(output, dim=0)

class TemporalLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TemporalLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timesteps, _ = x.size()
        output = torch.jit.annotate(List[torch.Tensor], [])

        for t in range(timesteps):
            timestep_output = self.linear(x[t])
            output.append(timestep_output)

        return torch.stack(output, dim=0)

class MuHopf(nn.Module):
    def __init__(self, input_dim: int, dt: float, num_steps: int,
                 min_omega: float, max_omega: float):

        super(MuHopf, self).__init__()
        self.omegas = nn.Parameter(torch.rand(input_dim))
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt
        self.mu0 = 2.0
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor, r: torch.Tensor,
                phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert r.size() == x.size()
        assert x.size() == phi.size()

        output = torch.jit.annotate(List[torch.Tensor], [])

        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2*torch.pi)

        for t in range(self.num_steps):
            mu = (self.mu0 + 5*x)
            r = r + ((mu - (r**2)) * r) * self.dt
            phi = phi + omegas * self.dt
            z_r = r * torch.cos(phi)
            z_i = r * torch.sin(phi)
            z_t = torch.complex(z_r, z_i)
            output.append(z_t)

        return torch.stack(output, dim=0), r, phi

class ResHopf(torch.nn.Module):
    
    def __init__(self, units, min_omega, max_omega, dt, 
                 num_steps, train_omegas=True, 
                 input_scaler=5.0, *args, **kwargs) -> None:
        
        super(ResHopf, self).__init__(*args, **kwargs)
        
        self.units = units
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt
        self.num_steps = num_steps

        self.mu0 = torch.tensor(1.0)
        self.beta1 = 1.0
        self.input_scaler = input_scaler
        self.train_omegas = train_omegas

        self.omegas = torch.autograd.Variable(torch.randn((1, units)), 
                                              requires_grad=self.train_omegas)

    def forward(self, X, r, phi):
        
        Xr = torch.real(X)
        Xi = torch.imag(X)
        assert Xr.shape == Xi.shape
        assert r.shape == Xr.shape
        assert r.shape == phi.shape 
        
        r_save = torch.zeros(Xr.shape)
        phi_save = torch.zeros(Xr.shape)
        
        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2*torch.pi)
        
        for t in range(self.num_steps):
            input_r = self.input_scaler*Xr[t]*torch.cos(phi)
            input_phi = self.input_scaler*Xi[t]*torch.sin(phi)
            r = r + ((self.mu0 - self.beta1*(r**2)) * r + input_r) * self.dt
            phi = phi + (omegas - input_phi) * self.dt            
            r_save[t] = r
            phi_save[t] = phi
        
        z_r = r_save * torch.cos(phi_save)
        z_i = r_save * torch.sin(phi_save)

        return torch.complex(z_r, z_i), r, phi

class SingledtResHopf(torch.nn.Module):
    
    def __init__(self, units, min_omega, max_omega, dt, 
                 train_omegas=True, 
                 input_scaler=5.0, *args, **kwargs) -> None:

        super(SingledtResHopf, self).__init__(*args, **kwargs)

        self.units = units
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt

        self.mu0 = torch.tensor(1.0)
        self.beta1 = 1.0
        self.input_scaler = input_scaler
        self.train_omegas = train_omegas

        self.omegas = torch.autograd.Variable(torch.randn((1, units)), 
                                              requires_grad=self.train_omegas)

    def forward(self, X, r, phi):
        
        Xr = torch.real(X)
        Xi = torch.imag(X)
        assert Xr.shape == Xi.shape
        assert r.shape == Xr.shape
        assert r.shape == phi.shape 
        
        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2*torch.pi)
        
        input_r = self.input_scaler*Xr*torch.cos(phi)
        input_phi = self.input_scaler*Xi*torch.sin(phi)
        r = r + ((self.mu0 - self.beta1*(r**2)) * r + input_r) * self.dt
        phi = phi + (omegas - input_phi) * self.dt            

        z_r = r * torch.cos(phi)
        z_i = r * torch.sin(phi)

        return torch.complex(z_r, z_i), r, phi

class FF(nn.Module):

    def __init__(self, input_units, units):
        super(FF, self).__init__()
        self.units = units
        self.state_size = units
        self.j_h = nn.Linear(units, units)
        self.j_x = nn.Linear(input_units, units)
        self.k_h = nn.Linear(units, units)
        self.k_x = nn.Linear(input_units, units)

    def forward(self, inputs, states):
        prev_output = states[0]
        j = torch.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = torch.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output] 