/*
 * Steady-state 2D Darcy flow on the unit square using P1 conforming FEM.
 *
 *   -div(k(x,y) grad u) = f(x,y)   in (0,1)x(0,1)
 *   u = 0                            on boundary
 *
 *   k(x,y) = 10 inside circle centered at (0.5,0.5) radius 0.2,
 *            1  elsewhere.
 *   f(x,y) = exp(-50*((x-0.5)^2 + (y-0.5)^2)).
 *
 * Implementation choice:
 *   We construct a structured triangular mesh of the unit square manually
 *   (each cell of an NxN grid split into 2 triangles). We then assemble the
 *   global stiffness matrix A and right-hand side b by element loops with
 *   quadrature, using P1 (linear) Lagrange basis functions on each triangle.
 *   The Dirichlet condition u=0 on the boundary is imposed strongly via
 *   MatZeroRows.
 *
 *   The linear system A u = b is solved with PETSc KSP (defaults configurable
 *   from the command line, e.g. -ksp_type cg -pc_type gamg).
 *
 *   After solving, we compute and print the L2 norm of u, sqrt(u^T M u),
 *   where M is the FE mass matrix.
 */

#include <petsc.h>
#include <math.h>

static inline PetscScalar permeability(PetscReal x, PetscReal y)
{
  PetscReal dx = x - 0.5, dy = y - 0.5;
  PetscReal r2 = dx*dx + dy*dy;
  return (r2 <= 0.2*0.2) ? 10.0 : 1.0;
}

static inline PetscScalar source(PetscReal x, PetscReal y)
{
  PetscReal dx = x - 0.5, dy = y - 0.5;
  return PetscExpReal(-50.0 * (dx*dx + dy*dy));
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscInt       N = 64; /* number of subdivisions per side */
  ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL); CHKERRQ(ierr);

  PetscMPIInt size;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example is sequential.");

  /* Mesh data:
   *   Nv = (N+1)*(N+1) vertices,
   *   Nt = 2*N*N triangles.
   * Vertex (i,j) has index i + j*(N+1), coordinates (i*h, j*h), h=1/N.
   * For each cell (i,j) with i,j in [0,N-1], two triangles:
   *   T1: (i,j), (i+1,j), (i+1,j+1)
   *   T2: (i,j), (i+1,j+1), (i,j+1)
   */
  PetscInt Nv = (N+1)*(N+1);
  PetscReal h = 1.0/(PetscReal)N;

  Mat A, M;
  Vec b, u;
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Nv, Nv); CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 9, NULL); CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &M); CHKERRQ(ierr);
  ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, Nv, Nv); CHKERRQ(ierr);
  ierr = MatSetType(M, MATAIJ); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(M, 9, NULL); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
  ierr = VecSetSizes(b, PETSC_DECIDE, Nv); CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  ierr = VecDuplicate(b, &u); CHKERRQ(ierr);
  ierr = VecSet(b, 0.0); CHKERRQ(ierr);

  /* 3-point quadrature for triangles (degree 2), barycentric:
   *   (2/3,1/6,1/6), (1/6,2/3,1/6), (1/6,1/6,2/3), weight = 1/3 each
   *   (these are weights on the reference triangle of area 1/2).
   */
  PetscReal qb[3][3] = {{2.0/3.0, 1.0/6.0, 1.0/6.0},
                        {1.0/6.0, 2.0/3.0, 1.0/6.0},
                        {1.0/6.0, 1.0/6.0, 2.0/3.0}};
  PetscReal qw[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};

  for (PetscInt j = 0; j < N; ++j) {
    for (PetscInt i = 0; i < N; ++i) {
      PetscInt v00 = i     + j*(N+1);
      PetscInt v10 = (i+1) + j*(N+1);
      PetscInt v11 = (i+1) + (j+1)*(N+1);
      PetscInt v01 = i     + (j+1)*(N+1);

      PetscInt tri[2][3] = { {v00, v10, v11}, {v00, v11, v01} };

      for (int t = 0; t < 2; ++t) {
        PetscInt *idx = tri[t];
        PetscReal x[3], y[3];
        for (int a = 0; a < 3; ++a) {
          PetscInt vid = idx[a];
          PetscInt vi = vid % (N+1);
          PetscInt vj = vid / (N+1);
          x[a] = vi*h;
          y[a] = vj*h;
        }

        /* Affine triangle: area and gradients of P1 basis */
        PetscReal x1=x[0], y1=y[0], x2=x[1], y2=y[1], x3=x[2], y3=y[2];
        PetscReal detJ = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
        PetscReal area = 0.5 * PetscAbsReal(detJ);

        /* grad(phi_a) constant per element:
         *   phi_1 = ((y2-y3)(x-x3) + (x3-x2)(y-y3)) / detJ
         *   etc.  We compute gradients directly.
         */
        PetscReal grad[3][2];
        grad[0][0] = (y2 - y3)/detJ; grad[0][1] = (x3 - x2)/detJ;
        grad[1][0] = (y3 - y1)/detJ; grad[1][1] = (x1 - x3)/detJ;
        grad[2][0] = (y1 - y2)/detJ; grad[2][1] = (x2 - x1)/detJ;

        PetscScalar Ke[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        PetscScalar Me[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        PetscScalar Fe[3] = {0,0,0};

        for (int q = 0; q < 3; ++q) {
          PetscReal L1 = qb[q][0], L2 = qb[q][1], L3 = qb[q][2];
          PetscReal xq = L1*x1 + L2*x2 + L3*x3;
          PetscReal yq = L1*y1 + L2*y2 + L3*y3;
          PetscReal w  = qw[q] * area; /* quadrature weight scaled by area */
          PetscScalar kq = permeability(xq, yq);
          PetscScalar fq = source(xq, yq);
          PetscReal phi[3] = {L1, L2, L3};

          for (int a = 0; a < 3; ++a) {
            Fe[a] += w * fq * phi[a];
            for (int bb = 0; bb < 3; ++bb) {
              Ke[a][bb] += w * kq * (grad[a][0]*grad[bb][0] + grad[a][1]*grad[bb][1]);
              Me[a][bb] += w * phi[a] * phi[bb];
            }
          }
        }

        ierr = MatSetValues(A, 3, idx, 3, idx, &Ke[0][0], ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValues(M, 3, idx, 3, idx, &Me[0][0], ADD_VALUES); CHKERRQ(ierr);
        ierr = VecSetValues(b, 3, idx, Fe, ADD_VALUES); CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  /* Apply Dirichlet u=0 on boundary vertices */
  PetscInt *bcrows;
  ierr = PetscMalloc1(4*(N+1), &bcrows); CHKERRQ(ierr);
  PetscInt nbc = 0;
  for (PetscInt i = 0; i <= N; ++i) {
    bcrows[nbc++] = i + 0*(N+1);     /* bottom */
    bcrows[nbc++] = i + N*(N+1);     /* top */
  }
  for (PetscInt j = 1; j < N; ++j) {
    bcrows[nbc++] = 0 + j*(N+1);     /* left */
    bcrows[nbc++] = N + j*(N+1);     /* right */
  }

  ierr = MatZeroRows(A, nbc, bcrows, 1.0, NULL, NULL); CHKERRQ(ierr);
  /* Set b on boundary rows to 0 */
  {
    PetscScalar *zeros;
    ierr = PetscCalloc1(nbc, &zeros); CHKERRQ(ierr);
    ierr = VecSetValues(b, nbc, bcrows, zeros, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
    ierr = PetscFree(zeros); CHKERRQ(ierr);
  }

  /* Solve */
  KSP ksp;
  PC  pc;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

  /* L2 norm: sqrt(u^T M u) */
  Vec Mu;
  ierr = VecDuplicate(u, &Mu); CHKERRQ(ierr);
  ierr = MatMult(M, u, Mu); CHKERRQ(ierr);
  PetscScalar dot;
  ierr = VecDot(u, Mu, &dot); CHKERRQ(ierr);
  PetscReal l2 = PetscSqrtReal(PetscRealPart(dot));
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%.15g\n", (double)l2); CHKERRQ(ierr);

  ierr = VecDestroy(&Mu); CHKERRQ(ierr);
  ierr = PetscFree(bcrows); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&u); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&M); CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
