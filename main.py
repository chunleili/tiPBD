if __name__ == "__main__":
    import taichi as ti
    from ui.parse_cli import parse_cli
    
    args = parse_cli()

    if args.use_dearpygui:
        import ui.dearpygui as gui
        gui.run()

    if args.use_webui:
        import ui.webui as webui
        webui.run()

    ti.init(**args.init_args)

    from  engine.solver_main import solver_main
    solver_main()