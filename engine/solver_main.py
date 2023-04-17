import taichi as ti

def solver_main():
    from engine.metadata import meta
    if 'constitutive_model' in meta.common:
        if meta.common['constitutive_model'] == 'arap':
            from engine.volumetric.arap import ARAP
            pbd_solver = ARAP()
        elif meta.common['constitutive_model'] == 'neohooken':
            from engine.volumetric.neohooken import NeoHooken
            pbd_solver = NeoHooken()
        else:
            raise NotImplementedError(f"constitutive_model {meta.common['constitutive_model']} not implemented")
    elif meta.materials[0]['type'] == 'fluid':
        from engine.fluid.pbf import Fluid
        pbd_solver = Fluid()
    
    from engine.visualize import GGUI, vis_sdf, vis_sparse_grid
    ggui = GGUI()

    meta.ggui = ggui

    meta.paused = meta.common["initial_pause"]

    if meta.get_common("use_sdf"):
        ggui.sdf_vertices = vis_sdf(pbd_solver.sdf.val, False)

    if meta.get_common("vis_sparse_grid"):
        ggui.sparse_grid_anchors, ggui.sparse_grid_indices = vis_sparse_grid(pbd_solver.sdf.val, pbd_solver.sdf.resolution)
        print("ggui.sparse_grid_anchors",ggui.sparse_grid_anchors)
        print("ggui.sparse_grid_indices",ggui.sparse_grid_indices)

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
                print("step: ", meta.step)
                meta.step+=1
                pbd_solver.substep()
                print("step once")

        #do the simulation in each step
        if not meta.paused:
            for _ in range(meta.num_substeps):
                pbd_solver.substep()
                # print("pbd_solver.mesh.mesh.verts.pos",pbd_solver.mesh.mesh.verts.pos)
            # if meta.use_multigrid:
                # coarse_to_fine()

 
        ggui.update(pos_show=pbd_solver.pos_show, indices_show=pbd_solver.indices_show)
        ggui.canvas.scene(ggui.scene)
        ggui.window.show()

        if meta.args.kernel_profiler:
            ti.profiler.print_kernel_profiler_info()

