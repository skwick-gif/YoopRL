# Nightly follow-up walk-forward runs (launch after the first long run finishes)
Set-Location -Path "D:\YoopRL"

# Wait ~6.5 hours (~23,400 seconds) before the second run
Start-Sleep -Seconds 23400
python -m backend.scripts.run_walk_forward --symbol TNA --config tmp/mini_config_v4_high_gamma.json --train-years 3 --test-years 1 --seed 5678 --output-dir backend/evaluation/nightly_long_windows_seed5678

# Wait another ~6.5 hours before the third run
Start-Sleep -Seconds 23400
python -m backend.scripts.run_walk_forward --symbol TNA --config tmp/mini_config_v4_high_gamma.json --train-years 3 --test-years 1 --seed 9012 --output-dir backend/evaluation/nightly_long_windows_seed9012
