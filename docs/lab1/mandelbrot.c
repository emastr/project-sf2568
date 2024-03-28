#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <mpi.h>

#define NX 2048 // number of columns
#define NY 2048 // number of rows
#define N_COLS 256 // number of colors

unsigned char compute_color(const double complex d, const double bound, const int n_cols) {
    int count = 1;
    double complex z = 0.0 + 0.0*I;

    while ((cabs(z) < bound) && (count < n_cols)) {
        z = z*z + d;
        count++;
    }

    return count;
}

int main(int argc, char **argv) {
    double start_time, end_time, total_time;

    
    // MPI stuff
    int rank, size;
    const int master_rank = 0;

    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Problem parameters
    const double bound = 1.;
    const double dx = 2.0*bound/(NX-1);   // horizontal step size 
    const double dy = 2.0*bound/(NY-1);   // vertical step size 

    int y_offset = rank*NY/size;        // offset in vertical coordinate

    double d_imag = 0., d_real = 0.;
    double complex d = 0.0 + 0.0*I;

    assert(NY % size == 0);

    // Allocate memory for block of matrix
    unsigned char color_block[NX*NY/size];

    // Iterate over row block size
    for (int j = 0; j < NY/size; j++) {

        // Each process computes one row block
        d_imag = (j + y_offset)*dy - bound;

        // Iterate over columns
        for (int i = 0; i < NX; i++) {
            d_real = i*dx - bound;

            d = d_real + d_imag*I;

            color_block[i+j*NX] = compute_color(d, bound, N_COLS);
        }
    }

    // Allocate memory for full color matrix
    unsigned char colors[NX*NY];

    MPI_Gather(color_block, NX*NY/size, MPI_UNSIGNED_CHAR, colors, NX*NY/size, MPI_UNSIGNED_CHAR, master_rank, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fp;
        fp = fopen("color.txt", "w");

        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                fprintf(fp, "%hhu ", colors[i+j*NX]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
    }

    end_time = MPI_Wtime();

    double elapsed_time = end_time - start_time;
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if ( rank == 0 ) {
        printf( "Total run time parallel: %f seconds\n", total_time);
    }
    
    MPI_Finalize();
    
    return 0;

}