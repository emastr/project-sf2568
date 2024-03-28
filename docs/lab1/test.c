#include <mpi.h>
#include <stdio.h>


int main(int argc, char **argv){
    int rank, size;
    MPI_Status status;
    char message[20];

    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    tag = 42;

    if (rank == 0){
        strcpy(message, "Hello, there");
        for (int i = 1; i < size; i++){
            rc = MPI_Send(message, 13, MPI_CHAR, i, tag, MPI_COMM_WORLD);
        }
    } else {
        rc = MPI_Recv(message, 13, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
    }
    printf("Rrank %d: %.13s\n", rank, message);
    rc = MPI_Finalize();
}

