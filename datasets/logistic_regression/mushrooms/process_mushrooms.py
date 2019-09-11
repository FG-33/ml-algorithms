import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help='Name of file', type=str)
args = parser.parse_args()


# Flags
# -f FILENAME
def main():
    # load data
    data = pd.read_csv(args.filename)

    # switch label column (first column) with last column
    cols = data.columns.tolist()
    cols = cols[1:] + cols[:1]
    # Remove col with a lot of '?' in it
    cols = cols[:10] + cols[11:]
    data = data[cols]

    # map features and label to numbers
    for i in data.columns.tolist():
        # get unique col values for mapping purposes
        col_uniq_vals = data[i].unique()
        # map col values to corresponding index in col_uniq_vals (need numbers not letters)
        map_to_vals = list(range(len(col_uniq_vals)))
        data[i] = data[i].replace(col_uniq_vals, map_to_vals)

    # normalize features [0,1]
    for i in data.columns.tolist()[:-1]:
        # get unique for mapping and calc mean
        col_uniq_vals = data[i].unique()
        mean_col = data[i].mean()

        if col_uniq_vals[-1] > 0:
            # apply mean normalization if no divided by zero
            map_to_vals = [round((j - mean_col) / col_uniq_vals[-1], 4) for j in col_uniq_vals]
            data[i] = data[i].replace(col_uniq_vals, map_to_vals)

    # write processed data to .txt
    data.to_csv(args.filename.split(".")[0] + "__processed.txt", sep=',', index=False, header=False)


if __name__ == "__main__":
    main()

""" unique values for each column
['x' 'b' 's' 'f' 'k' 'c']
['s' 'y' 'f' 'g']
['y' 'w' 'g' 'n' 'e' 'p' 'b' 'u' 'c' 'r']
['t' 'f']
['a' 'l' 'p' 'n' 'f' 'c' 'y' 's' 'm']
['f' 'a']
['c' 'w']
['b' 'n']
['k' 'n' 'g' 'p' 'w' 'h' 'u' 'e' 'b' 'r' 'y' 'o']
['e' 't']
['c' 'e' 'b' 'r' '?'] <- remove because there are a lot of entries with just '?'
['s' 'f' 'k' 'y']
['s' 'f' 'y' 'k']
['w' 'g' 'p' 'n' 'b' 'e' 'o' 'c' 'y']
['w' 'p' 'g' 'b' 'n' 'e' 'y' 'o' 'c']
['p']
['w' 'n' 'o' 'y']
['o' 't' 'n']
['p' 'e' 'l' 'f' 'n']
['n' 'k' 'u' 'h' 'w' 'r' 'o' 'y' 'b']
['n' 's' 'a' 'v' 'y' 'c']
['g' 'm' 'u' 'd' 'p' 'w' 'l']
['e' 'p']
"""