def filedialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    filename = filedialog.askopenfilename(initialdir="data/scene", title="Select a File")
    del root
    print("Open scene file: ", filename)
    return filename