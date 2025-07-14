def get_top_k(file_name,k=10):
    with open(file_name, 'r') as file:
            # Read all lines and store them in a list, removing any trailing newlines
            lines = [line.strip().split(":")[0].replace("_"," ").replace("-"," ") for line in file.readlines()]

    lines = lines[:k+1]

    return lines
