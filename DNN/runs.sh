#!/usr/bin/env bash

# Force the real Windows Python (not MSYS2 one!)
PYTHON="/c/Users/DELL/AppData/Local/Programs/Python/Python311/python.exe"

echo "Running STL models..."
# -----------------------------
# CASE: pglib_opf_case118_ieee
# -----------------------------

# input_dim = 2 * 118 = 236
# hidden_dims = [29, 59, 118, 236, 472, 944, 1888]

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 14 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 29 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 59 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 118 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 236 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 472 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 944 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 1888 --train_samples 25000

# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 250
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 1000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 1500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 2000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 2500
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 5000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 10000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 15000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 20000
# python3 main.py --model stl --case_name pglib_opf_case118_ieee --hidden_dims 3776 --train_samples 25000



# -----------------------------
# CASE: pglib_opf_case118_ieee (PLATEAU)
# -----------------------------
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 250
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 500
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 1000
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 1500
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 2000
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 2500
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 5000
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 10000
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 15000
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 20000
python3 main.py --model plateau --case_name pglib_opf_case118_ieee --train_samples 25000

echo "All runs completed!"