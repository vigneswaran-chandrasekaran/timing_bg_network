import torch
import numpy as np
import pandas as pd
import seaborn as sns
import ptitprince as pt
from scipy.stats import ttest_ind
# from donn import *
# from model_cond import BG
import matplotlib.pyplot as plt

# Hyperparameters
input_dim = 4           # example input dimensionality (adjust to what your BG expects)
action_dim = 3          # number of action channels on the GPi
units = 32              # number of complex units in the oscillator layers
dt = 0.01               # timestep in seconds
min_omega = 1.0
max_omega = 10.0

T_seconds = 2.56        # one revolution of a Libet clock = 2560 ms (conventional choice)
T = int(T_seconds / dt) # number of timesteps per trial
clock_ms = 2560.0       # ms per rotation

n_trials = 100          # trials per condition
threshold_M = 0.12         # GPi evidence threshold for detecting a movement (tune if needed)
threshold_W = 0.04

device = torch.device('cpu')

def step_to_ms(step_idx, T, total_ms=clock_ms):
    return (step_idx / (T - 1)) * total_ms

def step_to_deg(step_idx, T):
    return (step_idx / (T - 1)) * 360.0

def make_trial_input(T, input_dim, rng):
    t = np.linspace(0, 1.0, T)
    ramp_center = rng.uniform(0.3, 0.9)
    ramp_width = rng.uniform(0.05, 0.25)
    ramp = np.exp(-0.5 * ((t - ramp_center) / ramp_width) ** 2)
    
    base = rng.normal(0.0, 0.03, size=(T, input_dim))
    for ch in range(input_dim):
        scale = rng.uniform(0.5, 1.5)
        base[:, ch] += scale * ramp
    
    base = base.astype(np.float32)
    return base

current_seed = 81
rng = np.random.RandomState(current_seed)

models = {}
conditions = ['control', 'sc', 'pd']
colors = {'control': 'blue', 'sc': 'red', 'pd': 'green'}
labels = {'control': 'Control', 'sc': 'Schizophrenia', 'pd': 'Parkinson\'s'}

load = True

if not load:
    weights_path = f'wm_task_wts_1.pth'
    base_weights = torch.load(weights_path)
    for cond in conditions:
        
        if cond == 'control':
            bg = BG(input_dim, action_dim, units, dt, min_omega, max_omega, pd_flag=0, sc_flag=0)
        elif cond == 'sc':
            bg = BG(input_dim, action_dim, units, dt, min_omega, max_omega, pd_flag=0, sc_flag=1)
        elif cond == 'pd':
            bg = BG(input_dim, action_dim, units, dt, min_omega, max_omega, pd_flag=1, sc_flag=0)
        
        bg.load_state_dict(base_weights)
        bg.to(device)
        bg.eval()
        models[cond] = bg

    data = {c: {'gpi_traces': [], 'm_times': [], 'w_times': [], 'mw_diff': [], 'valid_traces': []} for c in conditions}

    for cond in conditions:
        bg = models[cond]
        for tr in range(n_trials):
            st = make_trial_input(T, input_dim, rng)
            state_tensor = torch.from_numpy(st).unsqueeze(1).to(device)
            
            with torch.no_grad():
                _, _, zipper = bg.forward(state_tensor)
            
            raw_gpi = zipper['gpi'].squeeze()
            max_gpi = np.max(raw_gpi, axis=1) if raw_gpi.ndim == 2 else raw_gpi
            data[cond]['gpi_traces'].append(max_gpi)
            
            w_idx = np.where(max_gpi >= threshold_W)[0]
            w_ms = (w_idx[0] / (T - 1)) * clock_ms if len(w_idx) > 0 else np.nan
            m_idx = np.where(max_gpi >= threshold_M)[0]
            m_ms = (m_idx[0] / (T - 1)) * clock_ms if len(m_idx) > 0 else np.nan

            data[cond]['w_times'].append(w_ms)
            data[cond]['m_times'].append(m_ms)

            if not np.isnan(m_ms) and not np.isnan(w_ms):
                data[cond]['mw_diff'].append(m_ms - w_ms)
                trace_aligned = np.full(T * 2, np.nan)
                start_insert = T - m_idx[0]
                trace_aligned[start_insert : start_insert + len(max_gpi)] = max_gpi
                data[cond]['valid_traces'].append(trace_aligned)
            else:
                data[cond]['mw_diff'].append(np.nan)

    m_sc = np.nanmean(data['sc']['mw_diff'])
    m_ct = np.nanmean(data['control']['mw_diff'])
    m_pd = np.nanmean(data['pd']['mw_diff'])
    print(f"Results -> SC: {m_sc:.2f}, Control: {m_ct:.2f}, PD: {m_pd:.2f}")

    save_path = f"wm_task_results_new.npz"

    np.savez_compressed(
        save_path,
        conditions=np.array(conditions, dtype=object),
        labels=np.array(labels, dtype=object),

        control_gpi=np.array(data['control']['gpi_traces'], dtype=object),
        control_m_times=np.array(data['control']['m_times'], dtype=object),
        control_w_times=np.array(data['control']['w_times'], dtype=object),
        control_mw_diff=np.array(data['control']['mw_diff'], dtype=object),
        control_valid_traces=np.array(data['control']['valid_traces'], dtype=object),

        sc_gpi=np.array(data['sc']['gpi_traces'], dtype=object),
        sc_m_times=np.array(data['sc']['m_times'], dtype=object),
        sc_w_times=np.array(data['sc']['w_times'], dtype=object),
        sc_mw_diff=np.array(data['sc']['mw_diff'], dtype=object),
        sc_valid_traces=np.array(data['sc']['valid_traces'], dtype=object),

        pd_gpi=np.array(data['pd']['gpi_traces'], dtype=object),
        pd_m_times=np.array(data['pd']['m_times'], dtype=object),
        pd_w_times=np.array(data['pd']['w_times'], dtype=object),
        pd_mw_diff=np.array(data['pd']['mw_diff'], dtype=object),
        pd_valid_traces=np.array(data['pd']['valid_traces'], dtype=object),
    )

    print(f"Saved results to {save_path}")

# Plot
plt.rcParams.update({'font.size': 12})
palette = {'control': 'green', 'sc': 'blue', 'pd': 'orange'}
labels = {'control': 'NC', 'sc': 'SCZ', 'pd': "PD"}
conditions = ['control', 'sc', 'pd']

# Prepare Data
data_list = []

if not load:
    for cond in conditions:
        valid_intervals = [d for d in data[cond]['mw_diff'] if not np.isnan(d)]
        for v in valid_intervals:
            data_list.append({
                'Condition': 'W–M Interval',
                'WM_Interval': v,
                'Group': cond
            })
else:
    npz_path = "wm_task_results_1.npz"

    z = np.load(npz_path, allow_pickle=True)
    for cond in conditions:
        mw_key = f"{cond}_mw_diff"
        mw_vals = z[mw_key]

        # Drop NaNs
        valid_intervals = [v for v in mw_vals if not np.isnan(v)]

        for v in valid_intervals:
            data_list.append({
                'Condition': 'W–M Interval',
                'WM_Interval': v,
                'Group': cond
            })



df = pd.DataFrame(data_list)

palette = {'control': 'green', 'sc': 'blue', 'pd': 'orange'}
group_order = ['control', 'sc', 'pd']
display_labels = ['NC', 'SCZ', 'PD'] 
palette_list = [palette[g] for g in group_order]

fig, ax = plt.subplots(figsize=(7, 5))

dx = "Group"
dy = "WM_Interval"
ort = "v"

cloud = pt.RainCloud(
    data=df, x=dx, y=dy, hue=dx,
    order=group_order,
    palette=palette, 
    orient=ort, ax=ax,
    width_viol=.5, width_box=.15,
    alpha=.6, dodge=False, 
    point_size=3, jitter=.05,
    box_showfliers=False
)

import matplotlib.patches as mpatches
patches = [p for p in ax.get_children() if isinstance(p, mpatches.PathPatch)]
for i, patch in enumerate(patches[:3]):
    patch.set_facecolor(palette_list[i])
    patch.set_edgecolor('gray')
    patch.set_alpha(0.6)

def add_sig(g1_idx, g2_idx, p_val, y_pos):
    if p_val < 0.05:
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
        ax.plot([g1_idx, g1_idx, g2_idx, g2_idx], [y_pos, y_pos + 3, y_pos + 3, y_pos], color='black', lw=1.2)
        ax.text((g1_idx + g2_idx) / 2, y_pos + 5, sig, ha='center', fontweight='bold')

max_y = df['WM_Interval'].max()
v_nc = df[df['Group'] == 'control']['WM_Interval']
v_scz = df[df['Group'] == 'sc']['WM_Interval']
v_pd = df[df['Group'] == 'pd']['WM_Interval']

_, p1 = ttest_ind(v_nc, v_scz, nan_policy='omit') 
add_sig(0, 1, p1, max_y)

_, p2 = ttest_ind(v_scz, v_pd, nan_policy='omit') 
add_sig(1, 2, p2, max_y + 10)

_, p3 = ttest_ind(v_nc, v_pd, nan_policy='omit')  
add_sig(0, 2, p3, max_y + 20)

ax.set_title("Distribution of M-W Intervals")
ax.set_ylabel("M-W Interval (ms)")
ax.set_xlabel("")

ax.set_xticklabels(display_labels)

if ax.get_legend():
    ax.get_legend().remove()
    
sns.despine()
plt.tight_layout()
plt.savefig("wm_interval_distribution.png", dpi=400, bbox_inches='tight')
plt.show()