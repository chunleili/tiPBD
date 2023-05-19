import taichi as ti


def solver_main():
    from engine.metadata import meta

    meta.is_standalone = False
    if meta.get_common("simulation_method") == "arap":
        from engine.volumetric.arap import ARAP as Solver

        pbd_solver = Solver()
    elif meta.get_common("simulation_method") == "neohooken":
        from engine.volumetric.neohooken import NeoHooken as Solver

        pbd_solver = Solver()
    elif meta.get_common("simulation_method") == "mass_spring_volumetric":
        from engine.volumetric.mass_spring_volumetric import MassSpring as Solver

        pbd_solver = Solver()
    elif meta.get_common("simulation_method") == "strain_based_dynamics":
        from engine.volumetric.strain_based_dynamics import StrainBasedDynamics as Solver

        pbd_solver = Solver()
    elif meta.get_common("simulation_method") == "arap_hpbd":
        from engine.volumetric.arap_hpbd import HPBD as Solver

        pbd_solver = Solver()

    elif meta.get_common("simulation_method") == "pbf":
        import engine.fluid.pbf as standalone_solver

        meta.is_standalone = True
        standalone_solver.main()
    elif meta.get_common("simulation_method") == "pbf2d":
        import engine.fluid.pbf2d_sparse as standalone_solver

        meta.is_standalone = True
        standalone_solver.main()
    elif meta.get_common("simulation_method") == "shape_matching_rigidbody":
        import engine.shape_matching.rigidbody as standalone_solver

        meta.is_standalone = True
        standalone_solver.main()
    elif meta.get_common("simulation_method") == "arap_multigrid":
        import engine.volumetric.arap_multigrid as standalone_solver

        meta.is_standalone = True
        standalone_solver.main()

    if meta.get_common("self_main", False) or meta.is_standalone or meta.args.no_json:
        return

    meta.pbd_solver = pbd_solver

    meta.paused = meta.get_common("initial_pause", False)
    meta.num_substeps = meta.get_common("num_substeps", 1)
    meta.frame = 0
    meta.step_num = 0

    # no gui mode
    meta.no_gui = meta.get_common("no_gui", False)
    if meta.no_gui:
        meta.max_frame = meta.get_common("max_frame", 1000)
        while meta.frame < meta.max_frame:
            for _ in range(meta.num_substeps):
                pbd_solver.substep()
                meta.iter = 0
                meta.step_num += 1
            meta.frame += 1
            print("frame", meta.frame)
        return

    from engine.visualize import GGUI, vis_sdf, vis_sparse_grid

    ggui = GGUI()
    meta.ggui = ggui
    indices_show = None
    if hasattr(pbd_solver, "indices_show"):
        indices_show = pbd_solver.indices_show
    if meta.get_common("use_sdf"):
        ggui.sdf_vertices = vis_sdf(pbd_solver.sdf.val, False)
    # if meta.get_common("vis_sparse_grid"):
    #     ggui.sparse_grid_anchors, ggui.sparse_grid_indices = vis_sparse_grid(pbd_solver.sdf.val, pbd_solver.sdf.resolution)
    #     print("ggui.sparse_grid_anchors",ggui.sparse_grid_anchors)
    #     print("ggui.sparse_grid_indices",ggui.sparse_grid_indices)
    meta.use_selector = meta.get_common("use_selector", default=False)

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
                meta.step_num += 1
            if e.key == "r":
                print("Reset")
                pbd_solver.__init__()
                meta.step_num = 0
                meta.frame = 0
            if e.key == "c":
                meta.selector.clear()
            if e.key == "i":
                meta.selected_ids = meta.selector.get_ids()
                print(meta.selected_ids)

        # step once coninuously
        if ggui.window.is_pressed("g"):
            print("Step once, step: ", meta.step_num)
            pbd_solver.substep()
            meta.step_num += 1

        ## initialize the selector
        if meta.use_selector and not hasattr(meta, "selector"):
            from ui.selector import Selector

            meta.selector = Selector(ggui.camera, ggui.window, pbd_solver.pos_show)
            meta.particle_per_vertex_color = meta.selector.per_vertex_color

        # use the selector
        if meta.use_selector:
            meta.selector.select()

        # do the simulation in each step
        if not meta.paused:
            for _ in range(meta.num_substeps):
                meta.iter = 0
                pbd_solver.substep()
                meta.step_num += 1
                # print("pbd_solver.mesh.mesh.verts.pos",pbd_solver.mesh.mesh.verts.pos)
            # if meta.use_multigrid:
            # coarse_to_fine()

        ggui.update(pos_show=pbd_solver.pos_show, indices_show=indices_show)
        ggui.canvas.scene(ggui.scene)
        meta.frame += 1
        ggui.window.show()

        if meta.args.kernel_profiler:
            ti.profiler.print_kernel_profiler_info()
