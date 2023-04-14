def filedialog():
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.filename = filedialog.askopenfilename(initialdir="data/scene", title="Select a File")
    path = root.filename
    print("Open scene file: ", path)
    return path