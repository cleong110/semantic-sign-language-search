from pathlib import Path

if __name__ == "__main__":
    npy_files = Path.cwd().rglob("*.npy")
    print(npy_files)

    for npy_file in npy_files:
        old_name = npy_file.name
        new_name = npy_file.name.replace("videos","")
        new_name = npy_file.name.replace("-using-model-asl-citizen","")

        # accidentally forgot the "s" in "videos" the first time
        #if old_name[0] =="s":
        #    new_name = old_name[1:]
        new_path = npy_file.with_name(new_name)

        print(npy_file)
        print(new_path)

        npy_file.rename(new_path)
