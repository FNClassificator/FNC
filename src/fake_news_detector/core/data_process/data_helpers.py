
""" 
Helpers functions to process the data 
on exploration and classification
"""

# Split data in three parts by value
def get_third_parts(min_i,max_i,data):
    third = (max_i - min_i) / 3
    th_one = min_i + third
    th_two = max_i - third
    first_f = 0
    second_f = 0
    third_f = 0
    for i in data:
        if i <= th_one:
            first_f += 1
        elif i >= th_two:
            third_f += 1
        else:
            second_f +=1
    return first_f, second_f, third_f