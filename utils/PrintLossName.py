def printLossName(lambCoeff):
    print('lambCoeff:', lambCoeff)
    my_dict = {}
    count = 0
    count_2 = 0
    count_3 = 0
    for i in lambCoeff:
        if i != 0:
            my_dict[f"index{count}"] = f'lamb{i}'
            count_2 += 1
        count += 1
    myString = ''
    
    for key, value in my_dict.items():
        myString += key
        myString += '_'
        myString += f'{value}'
        count_3 += 1
        if count_3 < count_2:
            myString += '_'
    return myString
