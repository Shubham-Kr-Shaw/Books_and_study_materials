list1 = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
list2 = [ 0, 1, 1, 0, 1, 2, 2, 0, 1]

combined_dict = {k:v for k,v in zip(list1, list2)}
sorted_dict = dict(sorted(combined_dict.items(), key=lambda item: item[1]))
print(item for item in sorted_dict)


# list3 = []
# list4 = []
# list5 = []
# list_final = []
# for i,item in enumerate(list2):
#     print
#     if item == 0:
#         # print(list1[i])
#         list3.append(list1[i])
#     elif item ==1:
#         list4.append(list1[i])
#     else:
#         list5.append(list1[i])

# list_final = list3 + list4 + list5

# print(list_final)



        

# Output :['a', 'd', 'h', 'b', 'c', 'e', 'i', 'f', 'g']

