import argparse
parser = argparse.ArgumentParser()

# -f FILENAME
parser.add_argument("-f", "--filename", help='Name of file', type=str)
parser.add_argument("-pre", "--prefixrows", help='How many rows before data begins', type=int)
args = parser.parse_args()


def main():
    result = ""
    with open(args.filename) as file:
        lines = [line.rstrip("\n") for line in file]
        lines = [",".join(line.split()[1:]) for line in lines[args.prefixrows:]]
        result = "\n".join(lines)

    with open(args.filename.split(".")[0] + "__processed.txt", "w") as file:
        file.write(result)


if __name__ == "__main__":
    main()
