import sys
import os
import time
from collections import defaultdict
import pandas as pd
import random


def calculate_error_prob(prob_list):
    if len(prob_list)==0:
        return 0
    if len(prob_list)==1:
        return prob_list[0]
    zero_error_prob = 1.00
    for x in range(len(prob_list)):
        zero_error_prob*=(1-prob_list[x])

    final_prob =0
    for x in range(len(prob_list)):
        final_prob+=((zero_error_prob/(1-prob_list[x]))*prob_list[x])

    return final_prob


def get_best_disk_greedy(prob_list, total_file_size ,factor):
    threshold = pow(10, -15)/factor
    new_prob_list=[x for x in prob_list]
    data_dict = defaultdict(list)
    for x in range(len(new_prob_list)):
        data_dict[new_prob_list[x]].append(x)
    error_prob_per_bit=calculate_error_prob(new_prob_list)/(total_file_size * 8)
    if error_prob_per_bit<=threshold or ((error_prob_per_bit-threshold)/threshold) <= .1:
        return []

    new_prob_list.sort(reverse=True)
    for x in range(len(new_prob_list)):
        error_prob_per_bit=calculate_error_prob(new_prob_list[x+1:])/(total_file_size * 8)
        if error_prob_per_bit<=threshold or ((error_prob_per_bit-threshold)/threshold)<=.1:
            selected_list=[]
            selected_probabilities=new_prob_list[:x+1]
            for number in selected_probabilities:
                selected_list.append(data_dict[number][0])
                data_dict[number].remove(data_dict[number][0])
            return selected_list



def calibration(data, train_pop, target_pop, sampled_train_pop, sampled_target_pop):
    calibrated_data = ((data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)) / (( (1 - data) * (1 - target_pop / train_pop) / (1 - sampled_target_pop/sampled_train_pop)) + (data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop))))
    return calibrated_data




def runner(improvement_factor):
    global output_file
    total_file_size = 0
    total_errors = 0
    file_size_list = []
    block_size = "1M"
    if file_type == "small":
        for _ in range(5000):
            file_size_list.append(random.randint(1, 1024))
            block_size = "4KB"
    elif file_type == "medium":
        for _ in range(200):
            file_size_list.append(random.randint(250, 750))
    else:
        for _ in range(20):
            file_size_list.append(random.randint(1024, 20*1024))
    file_name_list = []
    start = time.time()
    for file_size in file_size_list:
        epoch_time = int(time.time() * 1000)
        full_file_path = lustre_path + str(epoch_time)
        file_name_list.append (full_file_path)
        total_file_size += file_size
        print("creating file {} size: {} KB".format(full_file_path,file_size))
        command = "dd if=/dev/zero of=" + full_file_path + " bs=" + block_size + " count="+ str(file_size) + " >/dev/null 2>&1"
        os.system(command)
    if improvement_factor == 0: # dont run probabilistic model
        duration = "{:.1f}".format(time.time() - start)
        output_file.write("duration {} io overhead 0 captured error rate 0 \n".format(duration))
        output_file.flush()
        command = "rm " + lustre_path+ "* >/dev/null 2>&1"
        os.system(command)
        return 0, 1
    #print(random.randint(0,total_rows))
    selected_disks = []
    preds=[]
    for _ in file_size_list:
        random_server = random.randint(0,total_rows-1)
        selected_disks.append(df.iloc[random_server])
        preds.append(df.iloc[random_server][2])
        if selected_disks[-1][3] == 1:
            total_errors += 1

    if "ST4000DM000" in smart_file:
        t=[calibration(x, 1985684, 404, 808, 404) for x in preds]
    elif "ST12000NM0007" in smart_file:
        t=[calibration(x, 16321688, 1616, 3232,1616) for x in preds]
    prob_list=[]
    divisor = 20 * pow(10,9)
    for val in range(len(t)):
        prob_list.append(float((t[val] * file_size_list[val]) / divisor))

    checked_disks = get_best_disk_greedy(prob_list, total_file_size, improvement_factor)

    print ("Selected disks to check {}".format(checked_disks))
    total_checked_io_size = 0
    captured_errors = 0
    for checked_disk in checked_disks:
        print("reading file {} size: {} ".format(file_name_list[checked_disk], file_size_list[checked_disk]))
        command = "dd if=" + file_name_list[checked_disk] +" of=/dev/zero bs=" + block_size + " >/dev/null 2>&1"
        os.system(command)
        total_checked_io_size += file_size_list[checked_disk]
        if selected_disks[checked_disk][3] == 1:
            captured_errors +=1
    duration = "{:.1f}".format(time.time() - start)
    checked_io_percentage = total_checked_io_size/total_file_size
    captured_error_percentage = 1
    if total_errors != 0:
        captured_error_percentage = captured_errors/total_errors
    output_file.write("duration {} io overhead {} captured error rate {} \n"
                      .format(duration, checked_io_percentage,captured_error_percentage))
    output_file.flush()
    command = "rm " + lustre_path + "* >/dev/null 2>&1"
    os.system(command)
    return checked_io_percentage, captured_error_percentage


#improvement_factor = int(sys.argv[1])
improvement_factor = 2
repetitions = 1
file_type = "small"
lustre_path = "/expanse/lustre/scratch/earslan/temp_project/probabilistic/"

if len(sys.argv) > 1:
    lustre_path = sys.argv[1]
if len(sys.argv) > 2:
    smart_file = sys.argv[2]
if len(sys.argv) > 3:
    file_type = sys.argv[3]
if len(sys.argv) > 4:
    improvement_factor = int(sys.argv[4])
if len(sys.argv) > 5:
    repetitions = int(sys.argv[5])

log_file_name = "normal_"
if improvement_factor > 0:
    print("factor {}  rep {}".format(improvement_factor,repetitions))
    df = pd.read_csv(smart_file,header=None)
    total_rows = df.shape[0]
    log_file_name = "probabilistic_" + str(improvement_factor)+ "_"
log_file_name += file_type + ".txt"
output_file = open(log_file_name,"a+")


total_check_io_percentage = 0
total_captured_error_percentage = 0
for i in range(repetitions):
    checked_io_percentage, captured_error_percentage = runner(improvement_factor)
    total_check_io_percentage += checked_io_percentage
    total_captured_error_percentage += captured_error_percentage
print ("checked io percentage {} captured error percentage {}".format(total_check_io_percentage/repetitions,
                                                                      total_captured_error_percentage/repetitions))
#output_file.write("TOTAL IO percentage {} TOTAL error percentage {}\n"
#                  .format(total_check_io_percentage/repetitions, total_captured_error_percentage/repetitions))
output_file.flush()
output_file.close()
#checked_io_percentage, captured_error_percentage = runner(improvement_factor, run_probabilistic)

#print ("checked io percentage {} captured error percentage {}".format(checked_io_percentage,captured_error_percentage))
#runner(improvement_factor=10)
#runner(improvement_factor=100)
