#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>

double get_random(double sigma){
    double rand_uniform = (double)rand() / (double)RAND_MAX;
    return -sigma * log(1/rand_uniform - 1.);
}

void local_matmul(double complex* x, double complex* local_A, double complex* b, int n, int N) {
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < N; j++) {
            x[i] += local_A[i * N + j] * b[j];
        }
    }
}

void get_local_A(double complex* local_A, double* w, double* data_x, int n, int n_start, int N, int Ndata){
    double temp;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < N; j++){
            local_A[i*N+j] = 0.0;
            for (int k = 0; k < Ndata; k++){
                temp = (w[j] - w[n_start + i]) * data_x[k];
                local_A[i*N+j] += cos(temp) + I * sin(temp);
            }
        }
    }
}

void evaluate_fourier(double* y, double* x_eval, double complex* coef, double* w, int N, int Ndata){
    double tmp;
    for (int i = 0; i < Ndata; i++){
        y[i] = 0.0;
        for (int j = 0; j < N; j++){
            tmp = w[j] * x_eval[i];
            y[i] += creal(coef[j]) * cos(tmp) - cimag(coef[j]) * sin(tmp);
        }
    }
}

void get_local_b(double complex* local_b, double* w, double* data_x, double* data_y, int n, int n_start, int N, int Ndata){
    double temp;
    for (int i = 0; i < n; i++){
        local_b[i] = 0.0;
        for (int k = 0; k < Ndata; k++){
            temp = w[n_start + i] * data_x[k];
            local_b[i] += data_y[k] * (cos(temp) - I * sin(temp));
        }
    }
}

void complex_sum(void* in, void* inout, int* len, MPI_Datatype* dptr){
    double complex* in_data = in;
    double complex* inout_data = inout;
    for (int i = 0; i < *len; i++){
        inout_data[i] += in_data[i];
    }

}

MPI_Datatype get_mpi_complex_double(){
    MPI_Datatype MPI_COMPLEX_DOUBLE;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_COMPLEX_DOUBLE);
    MPI_Type_commit(&MPI_COMPLEX_DOUBLE);
    return MPI_COMPLEX_DOUBLE;
}

void ConjugateGradient(double complex* local_x, double complex* local_A, double complex* b, int n, int n_start, int* n_list, int* displs, int N, int Iters, int rank, int size){

    double total = 0.0;
    double complex rrold, rrnew, alpha, beta, pAp;
    double complex* local_r = (double complex*)malloc(n * sizeof(double complex));
    double complex* p = (double complex*)malloc(N * sizeof(double complex));
    double complex* local_p = (double complex*)malloc(n * sizeof(double complex));
    double complex* local_Ap = (double complex*)malloc(n * sizeof(double complex));

    MPI_Datatype MPI_COMPLEX_DOUBLE = get_mpi_complex_double();
    MPI_Op MPI_COMPLEX_SUM;
    MPI_Op_create(complex_sum, 1, &MPI_COMPLEX_SUM);

    for (int i = 0; i < N; i++){
        p[i] = b[i];
    }

    rrnew = 0.0;
    for (int i = 0; i < n; i++) {
        local_p[i] = p[n_start + i];
        local_r[i] = b[n_start + i];
        rrnew += conj(local_r[i]) * local_r[i];
        local_x[i] = 0.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &rrnew, 1, MPI_COMPLEX_DOUBLE, MPI_COMPLEX_SUM, MPI_COMM_WORLD);

    // main loop
    for (int m = 0; m < Iters; m++) {
        pAp = 0.0;
        local_matmul(local_Ap, local_A, p, n, N);


        for (int i = 0; i < n; i++){
            pAp += conj(local_p[i]) * local_Ap[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_COMPLEX_DOUBLE, MPI_COMPLEX_SUM, MPI_COMM_WORLD);

        alpha = rrnew / pAp;
        rrold = rrnew;

        rrnew = 0.0;
        for (int i = 0; i < n; i++) {
            local_x[i] += alpha * local_p[i];
            local_r[i] -= alpha * local_Ap[i];
            rrnew += conj(local_r[i]) * local_r[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &rrnew, 1, MPI_COMPLEX_DOUBLE, MPI_COMPLEX_SUM, MPI_COMM_WORLD);
        // fprintf(stderr, "rank %d: rrnew = %f\n", rank, rrnew);
        beta = rrnew/rrold;
        for (int i = 0; i < n; i++) {
            local_p[i] = local_r[i] + beta * local_p[i];
        }
        MPI_Allgatherv(local_p, n, MPI_COMPLEX_DOUBLE, p, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);
    }
}

void get_weight_proposal(double* weights_new, double* weights_old, double sigma, int N){
    for (int i = 0; i < N; i++){
        weights_new[i] = weights_old[i] + get_random(sigma);
    }
}

void accept_reject(double* weights_old, double* weights_new, double complex* local_coef, double complex* coef, double gamma, int N, int rank, int size){
    double abs_old;
    double abs_new;
    double rand_numb;
    for (int i = 0; i < N; i++){
        rand_numb = (double)rand() / (double)RAND_MAX;
        abs_new = abs(local_coef[i]);
        abs_old = abs(coef[i]);
        if (rand_numb < pow(abs_new/abs_old, gamma)){
            weights_old[i] = weights_new[i];
        }
    }

}

void load_data(char* path, double* data_x, double* data_y, int Ndata){
    FILE *inp;
    inp = fopen(path, "r");
    for (int i = 0; i < Ndata; i++)
    {
        fscanf(inp, "%lf %lf", &data_x[i], &data_y[i]);
    }
    fclose(inp);
}


int main(int argc, char** argv) {
    int rank, size, N, n, n_start, Ndata, Iters, N_metropolis;
    double gamma, sigma, sigma0;
    int info;
    FILE *inp;

    MPI_Init(&argc, &argv);
    MPI_Datatype MPI_COMPLEX_DOUBLE = get_mpi_complex_double();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //cfs/klemming/home/e/emastr/Public/conjgrad-project

    char data[] = "/cfs/klemming/home/v/vilmers/Private/Project/data.txt";
    char output[] = "/cfs/klemming/home/v/vilmers/Private/Project/run.txt";
    char model_path[] = "/cfs/klemming/home/v/vilmers/Private/Project/model.txt";

    sigma0 = 1.;
    sigma = 1.;
    Ndata = 200;
    N = 50;
    Iters = N-1;
    gamma = 2;
    N_metropolis = 1;

    int L = N/size;      // note: integer division
    int res = N%size;      // remainder
    bool Ri = (rank < res);
    n = L + Ri;

    n_start = rank*L + (rank < res ? rank : res);

    int* displs = (int*)malloc(size * sizeof(int));
    int* n_list = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++){
        n_list[i] = L + (i < res);
        displs[i] = i*L + (i < res ? i : res);
        //fprintf(stderr, "rank %d: n_list[%d] = %d, displs[%d] = %d\n", rank, i, n_list[i], i, displs[i]);
    }

    clock_t start, diff;
    int msec;
    if (rank == 0){
        start = clock(), diff;
    }
    double complex* local_A = (double complex*)malloc(n * N * sizeof(double complex));
    double complex* local_b = (double complex*)malloc(n * sizeof(double complex));
    double complex* b = (double complex*)malloc(N * sizeof(double complex));
    double* weights = (double*)malloc(N*sizeof(double));
    double* weights_old = (double*)malloc(N*sizeof(double));
    double* weights_new = (double*)malloc(N*sizeof(double));
    double* data_x = (double*)malloc(Ndata * sizeof(double));
    double* data_y = (double*)malloc(Ndata * sizeof(double));
    double complex* local_coef = (double complex*)malloc(n * sizeof(double complex));
    double* y = (double*)malloc(Ndata * sizeof(double));
    double complex* coef = (double complex*)malloc(N * sizeof(double complex));
    double complex* coef_new = (double complex*)malloc(N * sizeof(double complex));

    for (int i = 0; i < N; i++){
        weights[i] = 0;
        weights_old[i] = 0;
        weights_new[i] = 0;
    }


    load_data(data, data_x, data_y, Ndata);


    // Initial solve
    if (rank == 0){
            get_weight_proposal(weights_old, weights_old, sigma0, N); // 0.000035 s for K = 144
        }
    MPI_Bcast(weights_old, N, MPI_DOUBLE, 0, MPI_COMM_WORLD); // 0.00015 s for K = 144
    //get_weight_proposal(weights_old, weights_old, sigma0, N);
    get_local_A(local_A, weights_old, data_x, n, n_start, N, Ndata); // 0.005 for K=144, N=200
    get_local_b(local_b, weights_old, data_x, data_y, n, n_start, N, Ndata); // fraction of the above
    MPI_Allgatherv(local_b, n, MPI_COMPLEX_DOUBLE, b, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD); // 0.00007 for K=144
    ConjugateGradient(local_coef, local_A, b, n, n_start, n_list, displs, N, Iters, rank, size); // 0.008
    MPI_Allgatherv(local_coef, n, MPI_COMPLEX_DOUBLE, coef, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD); //0.00003

    for (int i; i < N_metropolis; i++){
        // Proposal step
        if (rank == 0){
            get_weight_proposal(weights_new, weights_old, sigma, N);
        }
        MPI_Bcast(weights_new, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        get_local_A(local_A, weights_new, data_x, n, n_start, N, Ndata);
        get_local_b(local_b, weights_new, data_x, data_y, n, n_start, N, Ndata);
        MPI_Allgatherv(local_b, n, MPI_COMPLEX_DOUBLE, b, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);
        ConjugateGradient(local_coef, local_A, b, n, n_start, n_list, displs, N, Iters, rank, size);
        MPI_Allgatherv(local_coef, n, MPI_COMPLEX_DOUBLE, coef_new, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);        // Take Step
        if (rank == 0){
            accept_reject(weights_old, weights_new, coef, coef_new, gamma, N, rank, size);
        }
        MPI_Bcast(weights_old, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Solve again
        get_local_A(local_A, weights_old, data_x, n, n_start, N, Ndata);
        get_local_b(local_b, weights_old, data_x, data_y, n, n_start, N, Ndata);
        MPI_Allgatherv(local_b, n, MPI_COMPLEX_DOUBLE, b, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);
        ConjugateGradient(local_coef, local_A, b, n, n_start, n_list, displs, N, Iters, rank, size);
        MPI_Allgatherv(local_coef, n, MPI_COMPLEX_DOUBLE, coef, n_list, displs, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);
    }



    if (rank == 0){
        diff = clock() - start;
        msec = diff * 1000 / CLOCKS_PER_SEC;
        evaluate_fourier(y, data_x, coef, weights_old, N, Ndata);
    }


    if (rank == 0) {
        for (int i = 0; i < Ndata; i++) {
            //printf("true %f, approx %f\n", data_y[i], y[i]);
        }

        for (int i = 0; i < N; i++) {
            //printf("x[%d] = %f+%fi, b[%d]=%f+%fi\n", i, coef[i], i, b[i]);
        }

        printf("Time taken %d seconds %d milliseconds \n", msec/1000, msec%1000);
        inp = fopen(output, "w");
        for (int i = 0; i < Ndata; i++) {
            fprintf(inp, "%f\n", y[i]);
        }
        fclose(inp);

        inp = fopen(model_path, "w");
        for (int i = 0; i < N; i++) {
            fprintf(inp, "%f  %f %f\n", weights_old[i], creal(coef[i]), cimag(coef[i]));
        }
        fclose(inp);

    }

    // 144 = 3*3*2*2 [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144]
    free(local_A);
    free(local_b);
    free(b);
    free(weights);
    free(data_x);
    free(data_y);
    free(local_coef);
    free(y);
    if (rank == 0){
        free(coef);
    }
    MPI_Finalize();
}
//end of code

