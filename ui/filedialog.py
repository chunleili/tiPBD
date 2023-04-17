def filedialog():
    from tkinter import filedialog
    filename = filedialog.askopenfilename(initialdir="data/scene", title="Select a File")
    print("Open scene file: ", filename)
    return filename