/*
  Steady-state 2D Darcy flow equation solver using PETSc FEM.

  Equation: -∇ · (k(x,y) ∇u(x,y)) = f(x,y) on the unit square (0,1)x(0,1).

  - u: pressure
  - k: permeability, piecewise constant
  - f: source term, Gaussian
  - Boundary Conditions: u = 0 on all boundaries (homogeneous Dirichlet).

  Discretization: Conforming P1 Lagrange Finite Elements on a quadrilateral mesh.

  Implementation Strategy:
  This program uses the PETSc DMPlex interface for unstructured meshes and finite element
  discretizations. DMPlex provides a powerful abstraction for managing mesh data and
  finite element spaces. The assembly of the linear system (A u = b) is automated
  by providing PETSc with the weak form of the PDE at the level of quadrature points.

  The weak form is: ∫ k(x,y) ∇v · ∇u dx dy = ∫ f(x,y) v dx dy for all test functions v.

  This is expressed to PETSc via 'Residual' and 'Jacobian' functions:
  - Residual F(u)_i = ∫ k ∇u · ∇v_i dx dy - ∫ f v_i dx dy
  - Jacobian J_ij = dF_i/du_j = ∫ k ∇v_j · ∇v_i dx dy

  We provide C functions (f0_u, f1_u, g3_uu) that define the integrands for these forms.
  PETSc handles the numerical quadrature and assembly into global matrix A and vector b.

  Boundary conditions are applied strongly by PETSc during the assembly process.

  The resulting linear system is solved using the KSP (Krylov Subspace) interface,
  allowing for flexible solver and preconditioner choices from the command line.
*/

#include <petsc.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscsnes.h> /* For DMPlexSNESCompute... functions which we use for linear assembly */

/* 
  Define the permeability field k(x,y).
  k = 10.0 inside a circle of radius 0.2 centered at (0.5, 0.5).
  k = 1.0 elsewhere.
*/
static PetscReal permeability_func(PetscReal x, PetscReal y)
{
  const PetscReal dx = x - 0.5;
  const PetscReal dy = y - 0.5;
  if (dx * dx + dy * dy < 0.2 * 0.2) return 10.0;
  return 1.0;
}

/* 
  Define the source term f(x,y).
  f is a Gaussian centered at (0.5, 0.5).
*/
static PetscReal source_func(PetscReal x, PetscReal y)
{
  const PetscReal dx = x - 0.5;
  const PetscReal dy = y - 0.5;
  return PetscExpReal(-50.0 * (dx * dx + dy * dy));
}

/*
  PETSc integrand for the source term part of the residual: ∫ f*v dx
  This function provides f(x,y) at a quadrature point x.
  f0[c] = f_c(x) where c is the component of the field.
  Here we have one scalar field, so c=0.
*/
static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = source_func(x[0], x[1]);
}

/*
  PETSc integrand for the diffusion term part of the residual: ∫ k ∇u · ∇v dx
  This function provides the term k*∇u at a quadrature point.
  f1[c*dim+d] = k * ∂u_c/∂x_d
  Here, c=0, dim=2.
*/
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal k = permeability_func(x[0], x[1]);
  for (PetscInt d = 0; d < dim; ++d) {
    f1[d] = k * u_x[d];
  }
}

/*
  PETSc integrand for the Jacobian of the diffusion term: ∫ k ∇w · ∇v dx
  This function provides the diffusion tensor k_ij at a quadrature point.
  For isotropic permeability k, the tensor is k*I, where I is the identity matrix.
  g3[c*dim*dim + f*dim*dim + i*dim + j] = ∂(k_{ij} ∂u_f/∂x_j)/∂(∂u_c/∂x_i)
  For our scalar problem, c=f=0. We provide the 2x2 tensor k*I.
*/
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal k = permeability_func(x[0], x[1]);
  for (PetscInt d = 0; d < dim; ++d) {
    g3[d * dim + d] = k;
  }
}

/*
  Function to define the Dirichlet boundary condition value.
  u = 0.0 on the boundary.
*/
static void bc_u0(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
}

/*
  PETSc integrand for computing the L2 norm of the solution.
  We want to compute ∫ u^2 dx.
*/
static void u_sq_integrand(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  *f0 = u[0] * u[0];
}

int main(int argc, char **argv)
{
  DM             dm;   /* Domain manager */
  PetscFE        fe;   /* Finite element */
  PetscDS        ds;   /* Discrete system */
  Mat            A;    /* System matrix (Jacobian) */
  Vec            b, u; /* Right-hand side and solution vectors */
  KSP            ksp;  /* Krylov solver */
  const PetscInt field = 0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, "2D Darcy Flow FEM Solver\n"));

  /* 1. Create the mesh (DM) */
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  /* 2. Setup the finite element space and discretization */
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 2, 1, PETSC_FALSE, 1, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Pressure"));
  PetscCall(DMSetField(dm, field, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));

  /* 3. Define the weak form of the PDE */
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, field, f0_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, field, field, NULL, NULL, NULL, g3_uu));

  /* 4. Define the boundary conditions (u=0 on all boundaries) */
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void))bc_u0, 1, &field, NULL));

  /* 5. Create matrix and vectors */
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Solution"));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "RHS"));
  PetscCall(DMCreateMatrix(dm, &A));

  /* 6. Assemble the linear system A u = b */
  /* For a linear problem, the Jacobian is the system matrix A */
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, u, A, A, NULL));
  /* For a linear problem, the residual at u=0 is F(0) = -b */
  PetscCall(VecSet(u, 0.0));
  PetscCall(DMPlexSNESComputeResidualFEM(dm, u, b, NULL));
  PetscCall(VecScale(b, -1.0));

  /* 7. Create and configure the KSP solver */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));

  /* 8. Solve the linear system */
  PetscCall(KSPSolve(ksp, b, u));

  /* 9. Compute the L2 norm of the solution */
  PetscScalar norm_sq;
  PetscCall(PetscDSSetObjective(ds, 0, &u_sq_integrand));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &norm_sq, NULL));
  PetscReal l2_norm = PetscSqrtReal(PetscRealPart(norm_sq));

  /* 10. Print the result - ONLY the numeric value */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%1.12e\n", (double)l2_norm));

  /* 11. Cleanup */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());

  return 0;
}
