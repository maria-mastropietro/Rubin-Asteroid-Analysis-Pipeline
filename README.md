# Asteroid Wrapper Analysis

## Download the parquet files from the Asteroid Institute
First authenticate with Google Cloud:
```bash
gcloud auth login
```

Then download the Rubin MPC parquet files:
```bash
gcloud storage rsync gs://asteroid-institute-public/production/rubin/mpc/obs_sbn/daily/ ./rubin_mpc/parquet --recursive --exclude='^((?!\.parquet$).)*$'
```

This will create the rubin_mpc folder.
The WrapperAnalysis folder must be saved in the same location as the rubin_mpc folder.

## Check the number of observations
Run: 
```bash
python3 check_length.py
```
This prints the number of observations in each parquet file.

## Run the preprocessing
Run:
```bash
python3 run_preprocessing.py
```
This script uses: Select_multiopp_split_plot.py and Select_numbered_split_plot.py.
In run_preprocessing.py, edit the REQUESTED list (line 14) and add the parquet file(s) you want to analyze. 
Example: 
```bash
REQUESTED = [ 
("2026-04-24", "obs_sbn_X05_2026-04-24.parquet"), 
("2026-04-27", "obs_sbn_X05_2026-04-27.parquet"), 
]
```
After starting the script, it will ask:
Run processing for {parquet_file(s)_number} non-zero parquet file(s)? [Y/N] 
Enter:
```bash
Y
```

The preprocessing step:
- classifies the asteroids into numbered asteroids and provisional asteroids;
- creates lightcurve plots;
- creates *_ALL.csv files.
Note: If the code is stopped before finishing, running it again will skip the objects already analyzed and continue from where it stopped.

## Check the epoch and observation requirements
Run:
```bash
python3 epochs.py
```
This checks for:
- objects with at least 3 epochs in 30-day epoch bins
- objects with at least 30 observations in at least 2 bands
It creates:
- epochs.txt
- found_objects.txt

## Split the selected objects
Run:
```bash
python3 split_found_objects.py
```
This splits the objects in found_objects.txt into:
- choose_physical.txt, for objects with total observations >= 165, where both the Fourier and LSM analyses can be done;
- choose_lsm.txt, for objects with total observations < 165, where only the LSM analysis can be done.

## Select NEAs and MBOs
Run:
```bash
python3 search_NEA_MBO.py
```
This comments out the asteroids that are neither NEAs nor MBOs in choose_physical.txt and choose_lsm.txt.

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

# NOTES:
## In Asteroid_physical_properties.py:
Use base_period_days_before_doubling to force the Fourier on one peak period:
```bash
best_days = (
        try_float(info.get("best_period_days"))
```
To not force it, use best_period_days:
```bash
    best_days = (
        try_float(info.get("best_period_days"))
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


