/*
  This program demonstrates the use of PETSc vectors (Vec) and scatter contexts (VecScatter).
  1. It creates a parallel PETSc vector (VECMPI) of size N.
  2. It initializes the vector elements to 1.0, 2.0, ..., N.
  3. It scales the vector by a factor of 2.0.
  4. It uses VecScatter to gather all elements of the parallel vector onto a
     sequential vector (VECSEQ) on MPI process 0.
  5. It prints the gathered vector on process 0.
*/

#include <petscvec.h>

static char help[] = "Creates a parallel vector, initializes it, scales it, and gathers it to process 0 for viewing.\n\n";

int main(int argc, char **argv) {
    Vec            x, y;
    PetscInt       N = 10;
    PetscMPIInt    rank;
    PetscInt       i, rstart, rend;
    PetscScalar    value;
    VecScatter     scatter_ctx;

    /* Initialize PETSc and MPI */
    PetscInitialize(&argc, &argv, (char*)0, help);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    /* Get the global vector size N from command-line options */
    PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL);

    /* Create a parallel vector x. PETSc will partition the vector among processes. */
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, N);
    VecSetFromOptions(x); /* Allows setting vector type, e.g., -vec_type mpi */

    /* Initialize vector x with values 1.0, 2.0, ..., N */
    /* Each process initializes its local part of the vector. */
    VecGetOwnershipRange(x, &rstart, &rend);
    for (i = rstart; i < rend; i++) {
        value = (PetscScalar)(i + 1.0);
        VecSetValues(x, 1, &i, &value, INSERT_VALUES);
    }
    /* Assemble the vector after all processes have set their local values. */
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    /* Scale the entire vector x by 2.0 */
    VecScale(x, 2.0);

    /* Create a scatter context to gather the vector to process 0. */
    /* VecScatterCreateToZero is a convenient routine that also creates the */
    /* sequential destination vector 'y' on rank 0. */
    VecScatterCreateToZero(x, &scatter_ctx, &y);

    /* Perform the scatter from the parallel vector x to the sequential vector y on rank 0. */
    VecScatterBegin(scatter_ctx, x, y, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter_ctx, x, y, INSERT_VALUES, SCATTER_FORWARD);

    /* On process 0, view the gathered vector y. */
    /* On other processes, y is a valid vector object but has size 0. */
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_SELF, "Gathered and scaled vector y on rank 0:\n");
        /* PETSC_VIEWER_STDOUT_SELF ensures only this process prints. */
        VecView(y, PETSC_VIEWER_STDOUT_SELF);
    }

    /* Clean up PETSc objects to avoid memory leaks */
    VecDestroy(&x);
    VecDestroy(&y);
    VecScatterDestroy(&scatter_ctx);

    /* Finalize PETSc */
    PetscFinalize();
    return 0;
}