def split_in_three(data_real, data_fake):
    min_v = min(data_fake.min(), data_real.min())
    max_v = max(data_fake.max(), data_real.max())

    tercio = (max_v - min_v) / 3
    # Calculate 1/3
    th_one = min_v + tercio
    # Calculate 2/3
    th_two = max_v - tercio

    first_f, second_f, third_f = split_data(th_one, th_two, data_fake)
    first_r, second_r, third_r = split_data(th_one, th_two, data_real)

    total_f =  len(data_fake)
    fake = [first_f/total_f, second_f/total_f, third_f/total_f]
    total_r =  len(data_real)
    real = [first_r/total_r, second_r/total_r, third_r/total_r]
    return fake, real

def split_data(th_one, th_two, data):
    first = 0
    second = 0
    third = 0
    for i in data:
        if i <= th_one:
            third += 1
        elif i >= th_two:
            first += 1
        else:
            second +=1
    return first, second, third