/*
  2D Rosenbrock function minimization with PETSc/Tao.

  Objective:
    f(x,y) = (1 - x)^2 + 100 (y - x^2)^2

  Build (example):
    mpicc rosenbrock2d_tao.c -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include \
      -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -o rosen

  Run (example):
    ./rosen -tao_type bnls -tao_monitor -tao_gatol 1e-10
*/

#include <petsctao.h>

static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *g;
  PetscScalar        x0, x1;
  PetscScalar        t1, t2;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(G, &g));

  x0 = x[0];
  x1 = x[1];

  t1 = (PetscScalar)1.0 - x0;          /* (1 - x) */
  t2 = x1 - x0 * x0;                   /* (y - x^2) */

  *f = (PetscReal)(t1 * t1 + (PetscScalar)100.0 * t2 * t2);

  /* Gradient:
       df/dx = -2(1-x) - 400 x (y - x^2)
       df/dy = 200 (y - x^2)
  */
  g[0] = (PetscScalar)(-2.0) * t1 - (PetscScalar)400.0 * x0 * t2;
  g[1] = (PetscScalar)200.0 * t2;

  PetscCall(VecRestoreArray(G, &g));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ctx)
{
  const PetscScalar *x;
  PetscScalar        x0, x1;
  PetscInt           rows[2] = {0, 1};
  PetscScalar        vals[4];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  x0 = x[0];
  x1 = x[1];

  /* Hessian:
       d2f/dx2 = 2 - 400(y - x^2) + 800 x^2 = 2 - 400 y + 1200 x^2
       d2f/dxdy = d2f/dydx = -400 x
       d2f/dy2 = 200
  */
  vals[0] = (PetscScalar)2.0 - (PetscScalar)400.0 * x1 + (PetscScalar)1200.0 * x0 * x0; /* (0,0) */
  vals[1] = (PetscScalar)(-400.0) * x0;                                                 /* (0,1) */
  vals[2] = (PetscScalar)(-400.0) * x0;                                                 /* (1,0) */
  vals[3] = (PetscScalar)200.0;                                                         /* (1,1) */

  PetscCall(MatZeroEntries(H));
  PetscCall(MatSetValues(H, 2, rows, 2, rows, vals, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

  if (Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatSetValues(Hpre, 2, rows, 2, rows, vals, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao        tao;
  Vec        x;
  Mat        H;
  PetscInt   n = 2;
  PetscScalar xinit[2] = {-1.2, 1.0};

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  /* Create solution vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetValues(x, 2, (PetscInt[]){0,1}, xinit, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /* Create Hessian matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
  PetscCall(MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(H));
  PetscCall(MatSetUp(H));

  /* Create Tao solver */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBNLS));
  PetscCall(TaoSetSolution(tao, x));

  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, NULL));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, NULL));

  PetscCall(TaoSetFromOptions(tao));

  /* Solve */
  PetscCall(TaoSolve(tao));

  /* Print final solution */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final solution (x):\n"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Cleanup */
  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}
