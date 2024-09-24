import os

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def list_subdirectory_sizes(base_directory,sizes):
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)
        if os.path.isdir(item_path):
            size = get_directory_size(item_path)
            # print(f"Directory: {item}, Size: {size/1024**3:.2f} GB")
            sizes.append((item, size))
            

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    sizes = []
    list_subdirectory_sizes(base_directory,sizes)
    sort = False
    if sort:
        sorted_sizes = sorted(sizes, key=lambda x: x[1], reverse=True)
        print("Sorted directories by size:")
        for name, size in sorted_sizes:
            print(f"Directory: {name}, Size: {size/1024**3:.2f} GB")
    else:
        for name, size in sizes:
            print(f"Directory: {name}, Size: {size/1024**3:.2f} GB")
    print(f"Total size: {sum(size for _, size in sizes)/1024**3:.2f} GB")