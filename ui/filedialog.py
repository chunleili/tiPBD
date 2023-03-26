def filedialog():
    """Ask for a filename to open.

    This returns an opened file path, or None if the dialog
    was cancelled. 
    """
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.filename = filedialog.askopenfilename(initialdir="data/scene", title="Select a File")
    path = root.filename
    print("Open file: ", path)
    return path