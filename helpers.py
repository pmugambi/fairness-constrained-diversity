def my_counter(l, value):
    count = 0
    for i in l:
        if i == value:
            count = count + 1
    return count


def list_of_lists_to_list(list_of_lists):
    l = []
    for i in list_of_lists:
        # print i
        l.append(i[0])
    # print l
    return l
