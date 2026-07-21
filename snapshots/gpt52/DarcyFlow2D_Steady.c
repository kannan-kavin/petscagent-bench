/*
Steady-state 2D Darcy flow on unit square with Dirichlet u=0 on boundary.

We use PETSc DMPlex + PetscFE to define a conforming continuous P1 (Lagrange) FEM space.
Assembly is done via PetscDS residual/Jacobian callbacks (automatic element integration).

PDE: -div(k grad u) = f
Weak form: \int k grad u · grad v = \int f v
Dirichlet: u=0 on all boundaries, imposed strongly via DMAddBoundary.

Permeability k(x,y): piecewise constant
  k=10 inside circle centered (0.5,0.5) radius 0.2, else 1.
Source f(x,y): exp(-50*((x-0.5)^2+(y-0.5)^2))

After solve, compute L2 norm of u over domain and print ONLY the numeric value.

Build (example):
  mpicc main.c -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include \
      -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -lm
Run (example):
  ./a.out -dm_plex_box_faces 32,32 -ksp_type cg -pc_type gamg
*/

#include <petscksp.h>
#include <petscdmplex.h>
#include <petscfe.h>

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[],
                 const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[],
                 const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[],
                 PetscScalar f0[])
{
  (void)dim; (void)Nf; (void)NfAux; (void)uOff; (void)uOff_x; (void)u; (void)u_t; (void)u_x;
  (void)aOff; (void)aOff_x; (void)a; (void)a_t; (void)a_x; (void)t; (void)numConstants; (void)constants;
  const PetscReal dx = x[0] - 0.5;
  const PetscReal dy = x[1] - 0.5;
  const PetscReal r2 = dx*dx + dy*dy;
  const PetscReal f  = PetscExpReal(-50.0 * r2);
  /* Residual form uses: F = \int ( f0*v + f1·grad(v) )
     For our PDE: \int (k grad u · grad v - f v) = 0
     => f0 = -f
  */
  f0[0] = -f;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[],
                 const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[],
                 const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[],
                 PetscScalar f1[])
{
  (void)Nf; (void)NfAux; (void)uOff; (void)uOff_x; (void)u; (void)u_t;
  (void)aOff; (void)aOff_x; (void)a; (void)a_t; (void)a_x; (void)t; (void)numConstants; (void)constants;
  const PetscReal dx = x[0] - 0.5;
  const PetscReal dy = x[1] - 0.5;
  const PetscReal r2 = dx*dx + dy*dy;
  const PetscReal k  = (r2 <= 0.2*0.2) ? 10.0 : 1.0;
  /* f1 = k * grad(u) */
  for (PetscInt d = 0; d < dim; ++d) f1[d] = k * u_x[uOff_x[0] + d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[],
                  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[],
                  const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[],
                  PetscScalar g3[])
{
  (void)Nf; (void)NfAux; (void)uOff; (void)uOff_x; (void)u; (void)u_t; (void)u_x;
  (void)aOff; (void)aOff_x; (void)a; (void)a_t; (void)a_x; (void)t; (void)u_tShift;
  (void)numConstants; (void)constants;
  const PetscReal dx = x[0] - 0.5;
  const PetscReal dy = x[1] - 0.5;
  const PetscReal r2 = dx*dx + dy*dy;
  const PetscReal k  = (r2 <= 0.2*0.2) ? 10.0 : 1.0;
  /* Jacobian block for grad terms: g3_{ij} = k * delta_{ij}
     Layout: g3[compI*dim*compJ*dim + i*dim + j] for scalar field => dim x dim.
  */
  for (PetscInt i = 0; i < dim; ++i) {
    for (PetscInt j = 0; j < dim; ++j) {
      g3[i*dim + j] = (i == j) ? k : 0.0;
    }
  }
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  /* Use a box mesh on (0,1)^2. Default is triangles unless -dm_plex_simplex 0 is given. */
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm)
{
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  PetscBool      hasLabel;
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_TRUE, NULL, PETSC_DEFAULT, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "u"));

  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));

  PetscCall(PetscDSSetResidual(ds, 0, f0_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));

  /* Strong Dirichlet u=0 on all boundary faces.
     DMPlexCreateBoxMesh marks boundary with label "marker" by default.
  */
  PetscCall(DMHasLabel(dm, "marker", &hasLabel));
  if (!hasLabel) {
    /* If absent, create boundary label from geometry. */
    PetscCall(DMCreateLabel(dm, "marker"));
    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
    PetscCall(DMPlexLabelComplete(dm, label));
  }
  PetscCall(DMGetLabel(dm, "marker", &label));

  /* Boundary ids for box mesh are typically 1..4 (or more). Apply to all ids present.
     We add a boundary condition that applies to any point with label value != 0 by using id=1
     and then setting the label values accordingly is messy; instead, add for common ids 1..8.
     Extra ids that do not exist are harmless.
  */
  for (PetscInt id = 1; id <= 8; ++id) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "dirichlet", label, 1, &id, 0, 0, NULL, (void(*)(void))NULL, NULL, NULL, NULL));
  }

  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeL2Norm(DM dm, Vec u, PetscReal *nrm)
{
  PetscFunctionBeginUser;
  PetscCall(VecNorm(u, NORM_2, nrm));
  /* VecNorm is not the FEM L2 norm over the domain; compute integral via PetscFE integration. */
  {
    PetscDS ds;
    PetscInt dim;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(DMGetDimension(dm, &dim));

    /* Use PetscDSComputeObjective with integrand = u^2.
       We implement objective integrand as g0 = u^2, no derivatives.
    */
    PetscErrorCode (*obj)(PetscInt,PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar*,void*);
    (void)obj;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective integrand for L2 norm squared: \int u^2 dx */
static void obj_u2(PetscInt dim, PetscReal time, const PetscReal x[],
                   PetscInt Nc, const PetscScalar u[], PetscScalar *obj, void *ctx)
{
  (void)dim; (void)time; (void)x; (void)ctx;
  /* Nc should be 1 */
  const PetscScalar val = u[0];
  obj[0] = val*val;
}

int main(int argc, char **argv)
{
  DM        dm;
  Mat       A;
  Vec       x, b;
  KSP       ksp;
  PetscReal l2sq = 0.0, l2 = 0.0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SetupDiscretization(dm));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(DMCreateMatrix(dm, &A));

  /* Assemble system */
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, x, A, A, NULL));
  PetscCall(DMPlexSNESComputeResidualFEM(dm, x, b, NULL));
  /* Residual routine returns F(u) = 0 form; at u=0, F = -b_rhs.
     We want A u = rhs, so set rhs = -F(0). */
  PetscCall(VecScale(b, -1.0));

  /* Apply Dirichlet strongly: PETSc FEM boundary handling is typically done in SNES.
     For a linear solve, we enforce by zeroing rows/cols for constrained dofs.
     We obtain constrained dofs from DM and use MatZeroRowsColumns.
  */
  {
    IS        is;
    PetscInt  n;
    PetscCall(DMGetStratumIS(dm, "marker", 1, &is));
    if (is) {
      /* Convert boundary points to dofs */
      IS dofis;
      PetscCall(DMPlexCreateSectionIndexIS(dm, is, &dofis));
      PetscCall(ISDestroy(&is));
      PetscCall(ISGetSize(dofis, &n));
      if (n > 0) {
        PetscCall(MatZeroRowsColumnsIS(A, dofis, 1.0, x, b));
      }
      PetscCall(ISDestroy(&dofis));
    } else {
      /* Fallback: try ids 2..8 */
      for (PetscInt id = 2; id <= 8; ++id) {
        PetscCall(DMGetStratumIS(dm, "marker", id, &is));
        if (!is) continue;
        IS dofis;
        PetscCall(DMPlexCreateSectionIndexIS(dm, is, &dofis));
        PetscCall(ISDestroy(&is));
        PetscCall(ISGetSize(dofis, &n));
        if (n > 0) PetscCall(MatZeroRowsColumnsIS(A, dofis, 1.0, x, b));
        PetscCall(ISDestroy(&dofis));
      }
    }
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  /* Compute FEM L2 norm: sqrt(\int u^2 dx) */
  {
    PetscDS ds;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetObjective(ds, 0, obj_u2));
    PetscCall(DMPlexComputeIntegralFEM(dm, x, &l2sq, NULL));
    l2 = PetscSqrtReal(l2sq);
  }

  /* Print only numeric value */
  {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (rank == 0) {
      PetscPrintf(PETSC_COMM_SELF, "%.16e\n", (double)l2);
    }
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}
