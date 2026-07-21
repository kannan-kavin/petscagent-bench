/*
  This program demonstrates how to use the PETSc/TAO library to solve
  the 2D Rosenbrock unconstrained minimization problem.

  The Rosenbrock function is a non-convex function used as a performance
  test problem for optimization algorithms.

  The function is defined as:
  f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

  The global minimum is at (x, y) = (1, 1), where f(1, 1) = 0.

  The gradient is:
  grad(f) = [ df/dx ] = [ -2(1-x) - 400x(y-x^2) ]
            [ df/dy ]   [      200(y-x^2)      ]
*/

#include <petsctao.h>

/*
  Function to compute the objective function value and the gradient.
  - tao: the TAO context
  - X: the current solution vector
  - f: pointer to store the objective function value
  - G: the gradient vector
  - ctx: user-defined context (not used here)
*/
PetscErrorCode FormObjectiveAndGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *g;

  PetscFunctionBeginUser;
  /* Get read-only access to the vector data */
  PetscCall(VecGetArrayRead(X, &x));
  /* Get write access to the vector data */
  PetscCall(VecGetArray(G, &g));

  /*
    Compute objective function
    f(x,y) = (1 - x)^2 + 100 * (y - x^2)^2
    In our vector, x is x[0] and y is x[1]
  */
  *f = (1.0 - x[0]) * (1.0 - x[0]) + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);

  /*
    Compute gradient
    df/dx = -2*(1-x) - 400*x*(y-x^2)
    df/dy = 200*(y-x^2)
  */
  g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
  g[1] = 200.0 * (x[1] - x[0] * x[0]);

  /* Restore access to the vectors */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec      x;    /* solution vector */
  Tao      tao;  /* TAO solver context */
  PetscInt n = 2; /* The dimension of the problem */

  /* Initialize PETSc and TAO */
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, "2D Rosenbrock Optimization\n"));

  /* Create a vector to store the solution */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));

  /* Set the initial guess. A common starting point for Rosenbrock is (-1.2, 1.0) */
  {
    PetscScalar *x_ptr;
    PetscCall(VecGetArray(x, &x_ptr));
    x_ptr[0] = -1.2;
    x_ptr[1] = 1.0;
    PetscCall(VecRestoreArray(x, &x_ptr));
  }

  /* Create the TAO solver */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  /* Set the solver type (e.g., blmvm, cg, nls). 
     TAOBLMVM is a good general-purpose choice for unconstrained problems. */
  PetscCall(TaoSetType(tao, TAOBLMVM));

  /* Set the solution vector */
  PetscCall(TaoSetSolution(tao, x));

  /* Set the routine for calculating the objective function and gradient */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveAndGradient, NULL));

  /* Allow user to set options from the command line */
  PetscCall(TaoSetFromOptions(tao));

  /* Solve the optimization problem */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solving Rosenbrock problem...\n"));
  PetscCall(TaoSolve(tao));

  /* Print the final solution */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--- Optimization Results ---\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final solution vector (should be close to [1, 1]):\n"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Check the solution */
  TaoConvergedReason reason;
  PetscCall(TaoGetConvergedReason(tao, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Termination reason: %s\n", TaoConvergedReasons[reason]));

  /* Clean up */
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());

  return 0;
}
