#%%
def read_file():
    data_array = []
    carfile = open('car/train.csv','r')
    for line in carfile:
        split_values = line.split(',')
        dict = {'buying':split_values[0],
                         'maint':split_values[1],
                         'doors':split_values[2],
                         'persons':split_values[3],
                         'lug_boot':split_values[4],
                         'safety':split_values[5],
                         'label':split_values[6]}
        data_array.append(dict)
    carfile.close()

read_file()


# %%
