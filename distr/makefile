c:
	mpicxx -O3 -o main main.cpp -lopenblas

cr:
	mpicxx -O3 -o bin_main main.cpp -lopenblas
# 	mpirun --oversubscribe -np 10 ./bin_main 4 4 5 5 Ax rrp
	mpirun --oversubscribe -np 10 ./bin_main 475 475 475 475 Ax rrp

cp:
	CC -O3 -o main main.cpp