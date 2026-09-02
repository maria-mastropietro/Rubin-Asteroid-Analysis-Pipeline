# RUBIN-ASTEROID-ANALYSIS-PIPELINE

## Download the parquet files from the Asteroid Institute
First authenticate with Google Cloud:
```bash
gcloud auth login
```

Then download the Rubin MPC parquet files:
```bash
gcloud storage rsync gs://asteroid-institute-public/production/rubin/mpc/obs_sbn/daily/ ./rubin_mpc/parquet --recursive --exclude='^((?!\.parquet$).)*$'
```

This will create the rubin_mpc folder.<br>
The Rubin-Asteroid-Analysis-Pipeline-main folder must be saved in the same location as the rubin_mpc folder.

## Alternative to download the parquet files from the Asteroid Institute
Alternatively, the parquet files can be downloaded manually from the [Asteroid Institute Rubin MPC downloads](https://b612.ai/rubin-mpc-downloads/). <br>
From the download page, select the Daily Partitions parquet file corresponding to the date you want to analyze.

If the files are downloaded manually, the folder structure must be created as:

```text
rubin_mpc/
└── parquet/
    └── YYYY-MM-DD/
        └── parquet/
            └── obs_sbn_X05_YYYY-MM-DD.parquet
Rubin-Asteroid-Analysis-Pipeline-main/
└──Asteroid_LSM.py
└──epochs.py
...
```

The date folder must match the date in the parquet filename.

The required folder structure can also be created from the terminal. Open the termina in your project directory.

**Linux:**
```bash
mkdir -p rubin_mpc/parquet/YYYY-MM-DD/parquet
mv obs_sbn_X05_YYYY-MM-DD.parquet rubin_mpc/parquet/YYYY-MM-DD/parquet/
```

**Windows Command Prompt:**
```bat
mkdir rubin_mpc\parquet\YYYY-MM-DD\parquet
move obs_sbn_X05_YYYY-MM-DD.parquet rubin_mpc\parquet\YYYY-MM-DD\parquet\
```

Replace `YYYY-MM-DD` with the date of the parquet file.

## Create a virtual environment
```bash
python3 -m venv env_asteroid
```

Activate it:

Linux/macOS
```bash
source env_asteroid/bin/activate
```

Windows
```bash
env_asteroid\Scripts\activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Check the number of observations
Run: 
```bash
python3 check_length.py
```
This prints the number of observations in each parquet file.

## Run the preprocessing
In run_preprocessing.py, edit the REQUESTED list (line 14) and add the parquet file(s) you want to analyze.<br>

Example: 
```bash
REQUESTED = [ 
("2026-02-27", "obs_sbn_X05_2026-02-27.parquet"), 
]
```

Run:
```bash
python3 run_preprocessing.py
```

After starting the script, it will ask:<br>
Run processing for ```{parquet_file(s)_number} non-zero parquet file(s)? [Y/N]```<br>
Enter:
```bash
Y
```

The preprocessing step:<br>
- classifies the asteroids into numbered asteroids and provisional asteroids;<br>
- creates lightcurve plots;<br>
- creates *_ALL.csv files.<br>

Note:
If the code is stopped before finishing, running it again will skip the objects already analyzed and continue from where it stopped.

## Check the epoch and observation requirements
Before running epochs.py, check the value of:

```python
THRESHOLD = 30
```
The default value used in the analysis is 30, meaning that an object must have at least 30 observations in at least 2 bands.

In some cases, a lower threshold can be useful. For example, for the objects in the 2026-02-27 parquet file, using:
```python
THRESHOLD = 10
```
is acceptable.

If THRESHOLD = 30 returns 0 objects, try running the analysis again with THRESHOLD = 10.

Then run:
```bash
python3 epochs.py
```
This checks for:<br>
- objects with at least 3 epochs in 30-day epoch bins<br>
- objects with at least 30 observations in at least 2 bands<br>

It creates:<br>
- epochs.txt<br>
- found_objects.txt

## Split the selected objects
Run:
```bash
python3 split_found_objects.py
```
This splits the objects in found_objects.txt into:<br>
- choose_physical.txt, for objects with total observations >= 165, where both the Fourier and LSM analyses can be done;<br>
- choose_lsm.txt, for objects with total observations < 165, where only the LSM analysis can be done.

## Select NEAs and MBOs
Run:
```bash
python3 search_NEA_MBO.py
```
This comments out the asteroids that are neither NEAs nor MBOs in choose_physical.txt and choose_lsm.txt.

## Run the period analyses
Before running the period-analysis scripts, check the contents of `choose_physical.txt` and `choose_lsm.txt`.

If one of these files contains 0 objects, do not run the corresponding analysis script.

For example:<br>
- if `choose_physical.txt` is empty, do not run `run_physical_properties_batch.py`;<br>
- if `choose_lsm.txt` is empty, do not run `run_LSM.py`.

## Run the physical-properties analysis
Run:
```bash
python3 run_physical_properties_batch.py
```
This runs the wrapper Asteroid_physical_properties.py for the objects in choose_physical.txt.

## Run the LSM-only analysis
Run:
```bash
python3 run_LSM.py
```
This runs the wrapper Asteroid_LSM.py for the objects in choose_lsm.txt.

# Notes:
## In Asteroid_physical_properties.py:
Use base_period_days_before_doubling to force the Fourier on one peak period:
```bash
best_days = (
    try_float(info.get("base_period_days_before_doubling "))
    ...
)
```
To not force it, use best_period_days:
```bash
best_days = (
    try_float(info.get("best_period_days"))
    ...
)
```

## In Period_search_High_Order_Fourier.py:
Use this to force the Fourier on one peak period:
```bash
#doubled = False
#reported_period = base_period
```

To not force it, use:
```bash
doubled = n_maxima == 1
reported_period = 2.0 * base_period if doubled else base_period
```


