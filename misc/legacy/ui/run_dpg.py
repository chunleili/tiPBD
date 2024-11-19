def run():
    from multiprocessing import Process

    sub_process = Process(target=run_dearpygui)
    sub_process.start()

def save_callback():
    print("Save Clicked")

def run_dearpygui():
    import dearpygui.dearpygui as dpg
    from math import sin

    dpg.create_context()

    data = []

    def update_plot_data(sender, app_data, plot_data):
        mouse_y = app_data[1]
        if len(plot_data) > 100:
            plot_data.pop(0)
        plot_data.append(sin(mouse_y / 30))
        dpg.set_value("mouse_y_lot", plot_data)

    def update_plot_data2(sender, app_data, plot_data):
        mouse_y = app_data[1]
        if len(plot_data) > 100:
            plot_data.pop(0)
        plot_data.append(sin(mouse_y / 30))
        dpg.set_value("plot2", plot_data)

    with dpg.window(label="Tutorial"):
        input_txt1 = dpg.add_text("Hello world")
        button1 = dpg.add_button(label="Save", callback=save_callback)
        input_txt2 = dpg.add_input_text(label="string")
        slider_float2 = dpg.add_slider_float(label="float")
        plot1 = dpg.add_simple_plot(label="mouse_y_lot", min_scale=0, max_scale=1.0, height=300, tag="mouse_y_lot")

        with dpg.plot(label="Plot Template", height=300) as plot2:
            dpg.add_plot_axis(dpg.mvXAxis, tag="x")
            dpg.add_plot_axis(dpg.mvYAxis, tag="y1")
            dpg.add_plot_axis(dpg.mvYAxis, tag="y2")
            dpg.add_line_series(x=[1, 2, 3, 4], y=[1, 4, 9, 16], parent="y1")
            dpg.add_line_series(x=[1, 2, 3, 4], y=[1, 2, 3, 4], parent="y2")

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=update_plot_data, user_data=data)

    def print_value(sender):
        print(dpg.get_value(sender))

    print(dpg.get_value(button1))
    print(dpg.get_value(input_txt2))
    print(dpg.get_value(slider_float2))
    dpg.set_item_callback(slider_float2, print_value)

    dpg.create_viewport()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    run_dearpygui()