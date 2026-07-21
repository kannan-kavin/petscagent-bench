/*
  PETSc test:
  - Create a VECMPI vector x of global size N (from -N, default 10)
  - Initialize x with values 1..N (global indexing)
  - Scale x by 2
  - Gather x onto rank 0 into a VECSEQ vector y using VecScatter
  - Print y on rank 0 using VecView

  Run example:
    mpiexec -n 3 ./main -N 10
*/

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N = 10;
  Vec            x = NULL,y = NULL;
  VecScatter     scatter = NULL;
  IS             isfrom = NULL,isto = NULL;
  PetscInt       rstart,rend,i;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);

  /* Create distributed vector x */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetType(x,VECMPI);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* Set local ownership range values: x[i] = i+1 (global i) */
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    PetscScalar v = (PetscScalar)(i + 1);
    ierr = VecSetValue(x,i,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* Scale by 2 */
  ierr = VecScale(x,2.0);CHKERRQ(ierr);

  /* Create a scatter that gathers all entries to rank 0 */
  if (rank == 0) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&y);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&isto);CHKERRQ(ierr);
  } else {
    /* y is not used on nonzero ranks, but VecScatterCreate requires a valid Vec */
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,&y);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&isto);CHKERRQ(ierr);
  }

  /* From-index set is the full global range on all ranks */
  ierr = ISCreateStride(PETSC_COMM_WORLD,N,0,1,&isfrom);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,isfrom,y,isto,&scatter);CHKERRQ(ierr);

  ierr = VecScatterBegin(scatter,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  if (rank == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Gathered y on rank 0 (should be 2,4,...,2N):\n");CHKERRQ(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&isfrom);CHKERRQ(ierr);
  ierr = ISDestroy(&isto);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
