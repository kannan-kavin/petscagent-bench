/* PETSc test:
 * - Create a VECMPI vector x of size N (default 10, override with -N).
 * - Initialize x with values 1.0, 2.0, ..., N.
 * - Scale x by 2.0.
 * - Use VecScatterCreateToZero to gather x into a VECSEQ vector y on rank 0.
 * - VecView y on rank 0.
 */

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x, y;
  VecScatter     scatter;
  PetscInt       N = 10;
  PetscInt       i, rstart, rend;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc, &argv, NULL, "PETSc VecScatter test\n"); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL); CHKERRQ(ierr);

  /* Create VECMPI vector x */
  ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
  ierr = VecSetSizes(x, PETSC_DECIDE, N); CHKERRQ(ierr);
  ierr = VecSetType(x, VECMPI); CHKERRQ(ierr);

  /* Initialize with values 1.0, 2.0, ..., N */
  ierr = VecGetOwnershipRange(x, &rstart, &rend); CHKERRQ(ierr);
  for (i = rstart; i < rend; i++) {
    PetscScalar v = (PetscScalar)(i + 1);
    ierr = VecSetValues(x, 1, &i, &v, INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

  /* Scale by 2.0 */
  ierr = VecScale(x, 2.0); CHKERRQ(ierr);

  /* Gather to rank 0 into a VECSEQ vector y */
  ierr = VecScatterCreateToZero(x, &scatter, &y); CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter, x, y, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, x, y, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  /* View y on rank 0 */
  if (rank == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Gathered vector y on rank 0:\n"); CHKERRQ(ierr);
    ierr = VecView(y, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&scatter); CHKERRQ(ierr);
  ierr = VecDestroy(&y); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
