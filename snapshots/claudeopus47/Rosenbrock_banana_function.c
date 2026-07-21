/*
 * 2D Rosenbrock Function minimization using PETSc/Tao
 * f(x,y) = (1 - x)^2 + 100 (y - x^2)^2
 * Minimum at (1, 1) with f = 0.
 */

#include <petsctao.h>

static char help[] = "Minimize the 2D Rosenbrock function using Tao.\n";

/* Evaluate objective and gradient */
static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *g;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(G, &g));

  PetscScalar xv = x[0];
  PetscScalar yv = x[1];

  *f = (1.0 - xv) * (1.0 - xv) + 100.0 * (yv - xv * xv) * (yv - xv * xv);

  g[0] = -2.0 * (1.0 - xv) - 400.0 * xv * (yv - xv * xv);
  g[1] = 200.0 * (yv - xv * xv);

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec       x;
  Tao       tao;
  PetscInt  n = 2;
  PetscScalar *xa;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Create the solution/variable vector */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));

  /* Set initial guess (classic Rosenbrock start) */
  PetscCall(VecGetArray(x, &xa));
  xa[0] = -1.2;
  xa[1] = 1.0;
  PetscCall(VecRestoreArray(x, &xa));

  /* Create Tao solver */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, NULL));
  PetscCall(TaoSetFromOptions(tao));

  /* Solve */
  PetscCall(TaoSolve(tao));

  /* Print the final solution */
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Final solution:\n"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));

  /* Clean up */
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}
