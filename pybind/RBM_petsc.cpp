// https://github.com/petsc/petsc/blob/6dd80bdec2980634f66887e5e77bea5a1e70dedf/src/mat/interface/matnull.c#L127

/*@
   MatNullSpaceCreateRigidBody - create rigid body modes from coordinates

   Collective on Vec

   Input Argument:
.  coords - block of coordinates of each node, must have block size set

   Output Argument:
.  sp - the null space

   Level: advanced

   Notes:
     If you are solving an elasticity problem you should likely use this, in conjunction with MatSetNearNullspace(), to provide information that 
     the PCGAMG preconditioner can use to construct a much more efficient preconditioner.

     If you are solving an elasticity problem with pure Neumann boundary conditions you can use this in conjunction with MatSetNullspace() to
     provide this information to the linear solver so it can handle the null space appropriately in the linear solution.


.seealso: MatNullSpaceCreate(), MatSetNearNullspace(), MatSetNullspace()
@*/
PetscErrorCode MatNullSpaceCreateRigidBody(Vec coords,MatNullSpace *sp)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *v[6],dots[5];
  Vec               vec[6];
  PetscInt          n,N,dim,nmodes,i,j;
  PetscReal         sN;

  PetscFunctionBegin;
  ierr = VecGetBlockSize(coords,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
  ierr = VecGetSize(coords,&N);CHKERRQ(ierr);
  n   /= dim;
  N   /= dim;
  sN = 1./PetscSqrtReal((PetscReal)N);
  switch (dim) {
  case 1:
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)coords),PETSC_TRUE,0,NULL,sp);CHKERRQ(ierr);
    break;
  case 2:
  case 3:
    nmodes = (dim == 2) ? 3 : 6;
    ierr   = VecCreate(PetscObjectComm((PetscObject)coords),&vec[0]);CHKERRQ(ierr);
    ierr   = VecSetSizes(vec[0],dim*n,dim*N);CHKERRQ(ierr);
    ierr   = VecSetBlockSize(vec[0],dim);CHKERRQ(ierr);
    ierr   = VecSetUp(vec[0]);CHKERRQ(ierr);
    for (i=1; i<nmodes; i++) {ierr = VecDuplicate(vec[0],&vec[i]);CHKERRQ(ierr);}
    for (i=0; i<nmodes; i++) {ierr = VecGetArray(vec[i],&v[i]);CHKERRQ(ierr);}
    ierr = VecGetArrayRead(coords,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (dim == 2) {
        v[0][i*2+0] = sN;
        v[0][i*2+1] = 0.;
        v[1][i*2+0] = 0.;
        v[1][i*2+1] = sN;
        /* Rotations */
        v[2][i*2+0] = -x[i*2+1];
        v[2][i*2+1] = x[i*2+0];
      } else {
        v[0][i*3+0] = sN;
        v[0][i*3+1] = 0.;
        v[0][i*3+2] = 0.;
        v[1][i*3+0] = 0.;
        v[1][i*3+1] = sN;
        v[1][i*3+2] = 0.;
        v[2][i*3+0] = 0.;
        v[2][i*3+1] = 0.;
        v[2][i*3+2] = sN;

        v[3][i*3+0] = x[i*3+1];
        v[3][i*3+1] = -x[i*3+0];
        v[3][i*3+2] = 0.;
        v[4][i*3+0] = 0.;
        v[4][i*3+1] = -x[i*3+2];
        v[4][i*3+2] = x[i*3+1];
        v[5][i*3+0] = x[i*3+2];
        v[5][i*3+1] = 0.;
        v[5][i*3+2] = -x[i*3+0];
      }
    }
    for (i=0; i<nmodes; i++) {ierr = VecRestoreArray(vec[i],&v[i]);CHKERRQ(ierr);}
    ierr = VecRestoreArrayRead(coords,&x);CHKERRQ(ierr);
    for (i=dim; i<nmodes; i++) {
      /* Orthonormalize vec[i] against vec[0:i-1] */
      ierr = VecMDot(vec[i],i,vec,dots);CHKERRQ(ierr);
      for (j=0; j<i; j++) dots[j] *= -1.;
      ierr = VecMAXPY(vec[i],i,dots,vec);CHKERRQ(ierr);
      ierr = VecNormalize(vec[i],NULL);CHKERRQ(ierr);
    }
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)coords),PETSC_FALSE,nmodes,vec,sp);CHKERRQ(ierr);
    for (i=0; i<nmodes; i++) {ierr = VecDestroy(&vec[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}