# make python environment variable
PYTHON=../venv/bin/python

c:
	mpicxx -O3 -o main main.cpp -lopenblas

cr:
	mpicxx -O3 -o bin_main main.cpp -lopenblas
	mpirun --oversubscribe -np 10 ./bin_main 4 4 5 5 Ax rrp
#	mpirun --oversubscribe -np 10 ./bin_main 475 475 475 475 Ax rrp

cp:
	CC -O3 -o main main.cpp

# Requires SLATE (github.com/icl-utk-edu/slate) + its blaspp/lapackpp deps.
# Point SLATE_DIR at an install prefix with include/ and lib/ (e.g. a NERSC
# "module load slate", or a site build), then `make baseline`.
#
# CUDA_INC/CUDA_LIB only matter because this SLATE build used
# gpu_backend=auto/cuda: its installed blas/device.hh unconditionally
# includes <cuda_runtime.h>, and libslate.so/libblaspp.so/liblapackpp.so
# have unresolved cuda*/cublas*/cusolver* symbols that must be satisfied at
# link time -- even for a CPU-only consumer like baseline.cpp that never
# exercises a GPU codepath at runtime (Target::HostTask is the default).
CUDA_INC ?= /usr/local/cuda-13.0/include
CUDA_LIB ?= /usr/local/cuda-13.0/lib64
baseline:
	mpicxx -O3 -std=c++17 -fopenmp -o baseline baseline.cpp \
		-I$(SLATE_DIR)/include -I$(CUDA_INC) -L$(SLATE_DIR)/lib -L$(CUDA_LIB) \
		-lslate -lblaspp -llapackpp -lopenblas -lcusolver -lcublas -lcudart \
		-Wl,-rpath,$(SLATE_DIR)/lib -Wl,-rpath,$(CUDA_LIB)

baseline_run:
	mpirun --oversubscribe -np 8 ./baseline 4 4 5 5

# NERSC Perlmutter build using Cray compiler wrappers (CC already links MPI
# + Cray LibSci). Assumes SLATE was built there with mpi=cray, blas=libsci,
# gpu_backend=none (see compile_perlmutter.sh), so -- unlike the `baseline`
# target above -- no CUDA include/link flags are needed.
baseline_perlmutter:
	CC -O3 -std=c++17 -fopenmp -o baseline baseline.cpp \
		-I$(SLATE_DIR)/include -L$(SLATE_DIR)/lib \
		-lslate -lblaspp -llapackpp \
		-Wl,-rpath,$(SLATE_DIR)/lib

plot_run:
	$(PYTHON) parse_runs.py runs/run4 --csv > runs/run4/results.csv
	$(PYTHON) plot.py runs/run4/results.csv