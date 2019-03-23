f = open("data/train_data.txt", "r").read()
f = f.replace("\n", " ")
keys = list(set(f.split(" ")))
keys.remove("")
keys = [int(x) for x in keys]
keys.sort()
encode = {key:num for num,key in enumerate(keys)}
decode = {num:key for num,key in enumerate(keys)}