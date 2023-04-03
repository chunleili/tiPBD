if __name__ == "__main__":
    from  engine.solver import Solver
    import taichi as ti
    from ui.parse_commandline_args import parse_commandline_args
    
    args = parse_commandline_args()

    if args.use_dearpygui:
        import ui.dearpygui as gui
        from multiprocessing import  Process
        gui_process = Process(target=gui.run)
        gui_process.start()
    
    ti.init(arch=args.arch, kernel_profiler=args.kernel_profiler, debug=args.debug)

    solver = Solver()
    solver.run()