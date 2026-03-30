from matplotlib import pyplot as plt
import numpy as np

nc_preds = np.load("predictions_0.npy")
pd_preds = np.load("predictions_1.npy")
sz_preds = np.load("predictions_2.npy")
print(nc_preds)
plt.plot([50, 100, 200, 400], np.std(nc_preds, axis=0), 
         label="Control", marker='o')

plt.plot([50, 100, 200, 400], np.std(pd_preds, axis=0), 
         label="Parkinson's", marker='o')

plt.plot([50, 100, 200, 400], np.std(sz_preds, axis=0), 
         label="Schizophrenia", marker='o')

plt.xlabel("Cue Length (timesteps)")
plt.ylabel("Standard Deviation of Predictions (timesteps)")
plt.title("Timing Variability under Different Conditions")
plt.legend()
plt.show()