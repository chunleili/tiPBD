import taichi as ti

def solver_main():
    from engine.metadata import meta
    if meta.common['constitutive_model'] == 'arap':
        from engine.fem.arap import ARAP
        pbd_solver = ARAP()
    elif meta.common['constitutive_model'] == 'neohooken':
        from engine.fem.neohooken import NeoHooken
        pbd_solver = NeoHooken()
    else:
        raise NotImplementedError
    
    from engine.visualize import GGUI
    ggui = GGUI()
    meta.paused = meta.common["initial_pause"]



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
                # debug_info(mesh.mesh.verts.pos)
                # debug_info(mesh.mesh.cells.lagrangian)
                print("step once")

        #do the simulation in each step
        if not meta.paused:
            for _ in range(meta.num_substeps):
                pbd_solver.substep()
                # print("pbd_solver.mesh.mesh.verts.pos",pbd_solver.mesh.mesh.verts.pos)
            # if meta.use_multigrid:
                # coarse_to_fine()
            
        ggui.update(pbd_solver.mesh.mesh.verts.pos, pbd_solver.mesh.surf_show)
    
        if meta.args.kernel_profiler:
            ti.profiler.print_kernel_profiler_info()


