if __name__ == "__main__":
    import taichi as ti
    from ui.parse_cli import parse_cli
    
    args = parse_cli()

    if args.use_dearpygui:
        import ui.dearpygui as gui
        from multiprocessing import  Process
        gui_process = Process(target=gui.run)
        gui_process.start()

    ti.init(arch=args.arch, kernel_profiler=args.kernel_profiler, debug=args.debug, device_memory_GB=args.device_memory_GB)

    args.use_solver_main = True
    if args.use_solver_main: # use the provided main in this lib
        from  engine.solver_main import solver_main
        solver_main()
    else: # manually give main
        import engine.fluid.pbf as solver
        solver.main()