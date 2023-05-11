if __name__ == "__main__":
    import taichi as ti
    import logging
    from ui.parse_cli import parse_cli
    
    args = parse_cli()
    logging.basicConfig(level=logging.INFO,
                        format=' %(levelname)s %(message)s')


    if args.use_dearpygui:
        import ui.dearpygui as dearpygui
        dearpygui.run()

    if args.use_webui:
        import ui.webui as webui
        webui.run()
    exit()
    ti.init(**args.init_args)

    from  engine.solver_main import solver_main
    solver_main()