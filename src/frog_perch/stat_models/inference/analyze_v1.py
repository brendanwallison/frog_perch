import os
import glob
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import io

BASE = "/home/breallis/dev/frog_perch/stan_results"
PATTERN = os.path.join(BASE, "call_intensity-20251210203435_*.csv")

csv_files = sorted(glob.glob(PATTERN))
print("Found CSVs:", csv_files)

# -------------------------------------------------------------------
# 1. Load CSVs manually (bypassing CmdStanPy)
# -------------------------------------------------------------------
import io
import os
import tempfile
import pandas as pd

def load_partial_csv(path):
    # Create a temporary file to accumulate valid rows
    tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    tmp_path = tmp.name

    header_written = False
    header = None

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            # Capture header once
            if not header_written and not (line[0].isdigit() or line[0] == "-"):
                header = line
                tmp.write(header)
                header_written = True
                continue

            # Keep only numeric draw rows
            if line[0].isdigit() or line[0] == "-":
                tmp.write(line)

    tmp.flush()
    tmp.close()

    # Now read the cleaned CSV from disk
    df = pd.read_csv(tmp_path)

    # Clean up temp file
    os.remove(tmp_path)

    return df

chains = [load_partial_csv(f) for f in csv_files]

# -------------------------------------------------------------------
# 2. Combine into ArviZ InferenceData
# -------------------------------------------------------------------
idata = az.from_dict(
    posterior={col: np.stack([c[col].values for c in chains], axis=0)
               for col in chains[0].columns}
)

print(idata)

# -------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------
print("\n==================== SAMPLER DIAGNOSTICS ====================")
print(fit.diagnose())

# -------------------------------------------------------------------
# Extract draws
# -------------------------------------------------------------------
draws = fit.draws_pd()   # tidy pandas dataframe

# Core parameters
u_draws = draws.filter(regex=r"^u\[\d+\]$")
gamma0 = draws["gamma_0"]
gamma1 = draws["gamma_1"]
sigma_day = draws["sigma_day"]

# Generated quantities
z_prob = draws.filter(regex=r"^z_prob\[\d+\]$")
day_intensity = draws.filter(regex=r"^day_intensity\[\d+\]$")

# -------------------------------------------------------------------
# 1. Plot daily intensity trajectory
# -------------------------------------------------------------------
plt.figure(figsize=(10, 5))
mean_intensity = day_intensity.mean(axis=0).values
lower = day_intensity.quantile(0.1).values
upper = day_intensity.quantile(0.9).values

days = np.arange(1, len(mean_intensity) + 1)

plt.plot(days, mean_intensity, label="Posterior mean")
plt.fill_between(days, lower, upper, alpha=0.3, label="80% interval")
plt.xlabel("Day index")
plt.ylabel("Daily call intensity (inv_logit(u[d] + gamma0))")
plt.title("Daily Call Intensity Trajectory")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 2. Time-of-day effect posterior
# -------------------------------------------------------------------
az.plot_posterior(idata, var_names=["gamma_0", "gamma_1"])
plt.suptitle("Time-of-Day Effect Posterior", fontsize=14)
plt.show()

# -------------------------------------------------------------------
# 3. Distribution of z_prob (posterior P(call | slice))
# -------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.hist(z_prob.values.flatten(), bins=50, alpha=0.7)
plt.xlabel("z_prob")
plt.ylabel("Frequency")
plt.title("Posterior Distribution of Slice-Level Call Probabilities")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 4. Trace plots for core parameters
# -------------------------------------------------------------------
az.plot_trace(idata, var_names=["sigma_day", "gamma_0", "gamma_1"])
plt.suptitle("Trace Plots for Key Parameters", fontsize=14)
plt.show()