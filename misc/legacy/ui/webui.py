def run():
    from multiprocessing import Process

    sub_process = Process(target=run_webui)
    sub_process.start()


def run_webui():
    import gradio as gr

    def greet(name):
        return "Hello123 " + name + "!"

    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch(inbrowser=True)
    print("webui launched")


if __name__ == "__main__":
    run_webui()
