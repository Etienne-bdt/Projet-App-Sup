import os
import csv

def register_images(csv_name, output_path="./", data_path="./") -> None:
    files: list[str] = os.listdir(data_path)
    with open(csv_name, "w") as f:
        writer =   csv.writer(f)
        writer.writerow(["img1", "img2", "mask"])
        prefixs: str = [f.split("_")[0] for f in files]
        prefixs = list(set([f for f in prefixs if "README" not in f]))
        print(prefixs)
        for i in range(len(prefixs)):
            files_prefixed = [f for f in files if prefixs[i] in f]
            mask = [f for f in files_prefixed if "cm" in f]
            img1 = [f for f in files_prefixed if "1" in f]
            img2 = [f for f in files_prefixed if "2" in f]
            writer.writerow([img1[0], img2[0], mask[0]])
    print("Image registered in CSV file")
    return None

def main():
    register_images("data.csv", data_path="./data")

if __name__ == "__main__":
    main()

    