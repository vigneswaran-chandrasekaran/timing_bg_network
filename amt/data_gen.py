import numpy as np

def generate_occlusion_data(screen_size=64,
                            num_steps=1000,
                            seed=None, force_trial=False, ttype=None):
    """
    Generates anticipation motion prediction data.
    
    - Block (width=5) moves right across 64-pixel screen
    - Two speeds: slow and fast (fast = 6 × slow)
    - Disappears at either 33% or 66% of the way
    - predicted_time ≈ close to 1000 (specifically: 960 for slow, 160 for fast)
    - predicted_time is THE SAME for 33% and 66% cutoff if speed is the same
    - predicted_time < 1000 always
    
    Returns:
        sequence: (1000, 64) numpy array
        predicted_time: int, when block would reach the end (if not occluded)
        speed: 'slow' or 'fast'
        cutoff_percent: 0.33 or 0.66
    """
    # if seed is not None:
    #     np.random.seed(seed)
    
    speed_chooser = np.random.randint(0, 2)
    cutoff_chooser = np.random.randint(0, 2)

    speed_list = ['slow', 'fast']
    cutoff_list = [0.33, 0.66]
    if not force_trial:
        # Choose speed and occlusion point
        speed_type = speed_list[speed_chooser]
        cutoff_percent = cutoff_list[cutoff_chooser]
    else:
        if ttype == 'slow_33':
            speed_type = 'slow'
            cutoff_percent = 0.33
        elif ttype == 'slow_66':
            speed_type = 'slow'
            cutoff_percent = 0.66
        elif ttype == 'fast_33':
            speed_type = 'fast'
            cutoff_percent = 0.33
        elif ttype == 'fast_66':
            speed_type = 'fast'
            cutoff_percent = 0.66

    cue_width = 5
    
    # Design: we want slow motion to take ~960 steps to cross 64 pixels
    base_steps_for_full_travel = 960
    pixels_per_step_slow = screen_size / base_steps_for_full_travel  # ≈ 0.0667 px/step
    
    if speed_type == 'slow':
        velocity = pixels_per_step_slow                    # ~0.0667 px/step → takes 960 steps
    else:
        velocity = pixels_per_step_slow * 6                # ~0.4 px/step → takes 160 steps
    
    # Total predicted time to cross full screen
    predicted_time = int(np.ceil(screen_size / velocity))
    predicted_time = min(predicted_time, num_steps - 1)  # safety
    
    # How far does it travel before occlusion?
    occlusion_distance = screen_size * cutoff_percent
    steps_visible = int(occlusion_distance / velocity)
    steps_visible = max(steps_visible, 10)  # at least some motion visible
    
    # Generate sequence
    sequence = []
    position = 0.0  # sub-pixel precision for smooth slow motion
    
    for t in range(num_steps):
        frame = np.ones(screen_size)
        
        if t < steps_visible:
            start = int(position)
            end = min(start + cue_width, screen_size)
            if start < screen_size:
                frame[start:end] = 5  # bright block
            position += velocity
        # after occlusion: stays dark
        
        sequence.append(frame)
    
    return np.array(sequence), predicted_time, speed_type, cutoff_percent

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
    
#     seq, pt, speed, cutoff = generate_occlusion_data(seed=42)
#     print(f"Predicted time: {pt}, Speed: {speed}, Cutoff: {cutoff}")
    
#     plt.imshow(seq.T, aspect='auto', cmap='gray_r')
#     plt.axvline(pt, color='red', linestyle='--', label='Predicted Time')
#     plt.title('Generated Occlusion Motion Prediction Sequence')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Screen Pixels')
#     plt.legend()
#     plt.show()
