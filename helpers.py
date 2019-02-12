def my_counter(l, value_list):
    count = 0
    for i in l:
        if i in value_list:
            count = count + 1
    return count


def list_of_lists_to_list(list_of_lists):
    l = []
    for i in list_of_lists:
        l.append(i[0])
    return l
