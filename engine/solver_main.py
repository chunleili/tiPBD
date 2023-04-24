import taichi as ti

def solver_main():
    from engine.metadata import meta #此处是第一次打开meta，因此也是打开filediag的地方
    if 'constitutive_model' in meta.common:
        if meta.common['constitutive_model'] == 'arap':
            from engine.volumetric.arap import ARAP
            pbd_solver = ARAP()
        elif meta.common['constitutive_model'] == 'neohooken':
            from engine.volumetric.neohooken import NeoHooken
            pbd_solver = NeoHooken()
        else:
            raise NotImplementedError(f"constitutive_model {meta.common['constitutive_model']} not implemented")
    elif meta.get_materials('type') == 'fluid':
        import engine.fluid.pbf as pbf
        pbf.main()
    elif meta.get_materials("type") == "fluid2d":
        import engine.fluid.pbf2d_sparse as pbf2d
        pbf2d.main()
    if meta.get_materials("type") == "shape_matching_rigidbody":
        import engine.shape_matching.rigidbody as rigidbody
        rigidbody.main()
    elif meta.get_materials("type") == "mass_spring_volumetric":
        from engine.volumetric.mass_spring_volumetric import MassSpring
        pbd_solver = MassSpring()
    elif meta.get_materials("type") == "strain_based_dynamics":
        from engine.volumetric.strain_based_dynamics import StrainBasedDynamics
        pbd_solver = StrainBasedDynamics()
    
    if meta.get_common("self_main", False):
        return

    from engine.visualize import GGUI, vis_sdf, vis_sparse_grid
    ggui = GGUI()

    meta.ggui = ggui

    meta.paused = meta.get_common("initial_pause", False)
    meta.num_substeps = meta.get_common("num_substeps", 1)

    if meta.get_common("use_sdf"):
        ggui.sdf_vertices = vis_sdf(pbd_solver.sdf.val, False)

    if meta.get_common("vis_sparse_grid"):
        ggui.sparse_grid_anchors, ggui.sparse_grid_indices = vis_sparse_grid(pbd_solver.sdf.val, pbd_solver.sdf.resolution)
        print("ggui.sparse_grid_anchors",ggui.sparse_grid_anchors)
        print("ggui.sparse_grid_indices",ggui.sparse_grid_indices)

    indices_show = None
    if hasattr(pbd_solver, "indices_show"):
        indices_show = pbd_solver.indices_show
    
    meta.frame=0
    meta.step_num=0
    # if meta.use_multigrid:
        # coarse_to_fine()
    while ggui.window.running:
        for e in ggui.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                meta.paused = not meta.paused
                print("paused:", meta.paused)
            if e.key == "f":
                print("Step once, step: ", meta.step_num)
                pbd_solver.substep()
                meta.step_num+=1 
        if ggui.window.is_pressed("g"):
            print("Step once, step: ", meta.step_num)
            pbd_solver.substep()
            meta.step_num+=1 

        #do the simulation in each step
        if not meta.paused:
            for _ in range(meta.num_substeps):
                meta.iter = 0
                pbd_solver.substep()
                meta.step_num+=1
                # print("pbd_solver.mesh.mesh.verts.pos",pbd_solver.mesh.mesh.verts.pos)
            # if meta.use_multigrid:
                # coarse_to_fine()

        ggui.update(pos_show=pbd_solver.pos_show, indices_show=indices_show)
        ggui.canvas.scene(ggui.scene)
        meta.frame += 1
        ggui.window.show()

        if meta.args.kernel_profiler:
            ti.profiler.print_kernel_profiler_info()