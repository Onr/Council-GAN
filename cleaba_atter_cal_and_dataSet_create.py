import os
from shutil import copyfile
from tqdm import tqdm
import argparse
argparse

parser = argparse.ArgumentParser()
parser.add_argument('--celeba_path', type=str, default='./datasets/CelebA', help='Path to celeba dataset.')
parser.add_argument('--celeba_anno_path', type=str, default='/Anno/list_attr_celeba.txt', help='path to celebe annotation file')
parser.add_argument('--celeba_path_list_eval_partition', type=str, default='/Eval/list_eval_partition.txt', help='path to celebe list_eval_partition')
parser.add_argument('--main_tag', type=str, default='Eyeglasses', help='main annotation to split celeba data by, one of: \'Male\' \'Blond_Hair\' \'Eyeglasses\' \'Gray_Hair\' \'Black_Hair\' \'Brown_Hair\'')
parser.add_argument('--sub_tag', type=str, default='Eyeglasses', help='sub annotation to split celeba data by, one of: \'Male\' \'Blond_Hair\' \'Eyeglasses\' \'Gray_Hair\' \'Black_Hair\' \'Brown_Hair\'')
opts = parser.parse_args()

path = opts.celeba_path
DataPath = os.path.join(path, 'Img/img_align_celeba')



f = open(path + opts.celeba_anno_path, "r")
f_eval = open(path + opts.celeba_path_list_eval_partition, "r")
num_of_images = int(f.readline())
tags = f.readline().split()
print('num of images: ' + str(num_of_images))
print('tags:')
print(str(tags))

tag_to_use_main = opts.main_tag
tag_to_use_sub = opts.sub_tag  # 'Male'  # 'Blond_Hair' # 'Eyeglasses' # 'Gray_Hair' # 'Black_Hair' #' Brown_Hair'
index_main = tags.index(tag_to_use_main) + 1
index_sub = tags.index(tag_to_use_sub) + 1


if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub)):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub))
if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main,'not_' + tag_to_use_sub)):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, 'not_' + tag_to_use_sub))

if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub)):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub))
if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main , 'not_' + tag_to_use_sub)):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, 'not_' + tag_to_use_sub))


if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub + '_test')):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub + '_test'))
if not os.path.exists(os.path.join(path, tag_to_use_main,tag_to_use_sub, tag_to_use_main, 'not_' + tag_to_use_sub + '_test')):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, 'not_' + tag_to_use_sub + '_test'))

if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub + '_test')):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub + '_test'))
if not os.path.exists(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, 'not_' + tag_to_use_sub + '_test')):
    os.makedirs(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, 'not_' + tag_to_use_sub + '_test'))


f_A_1 = open(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub, tag_to_use_main + '_' + tag_to_use_sub + ".txt"), "w")
f_A_2 = open(os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub, tag_to_use_main + '_not_' + tag_to_use_sub + ".txt"), "w")
f_B_1 = open(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main + '_' + tag_to_use_sub + ".txt"), "w")
f_B_2 = open(os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main + '_not_' + tag_to_use_sub + ".txt"), "w")

f_all_info = open(os.path.join(path,tag_to_use_main, tag_to_use_sub, tag_to_use_main + '_' + tag_to_use_sub + "_info.txt"), "w")

# print(index_main)
# print(index_sub)

numOf_A_1 = 0
numOf_A_2 = 0
numOf_B_1 = 0
numOf_B_2 = 0

for line in tqdm(f, total=num_of_images):
  curr_line = line.split()
  eval_partition = int(f_eval.readline().split()[1])
  addTest = ''
  add_eval = ''
  if eval_partition != 0:
      addTest = '_test'
      add_eval = str(eval_partition) + '_'

  fileEnder = 'png'
  file_name = curr_line[0].split('.')[0] + '.' + fileEnder
  srcFile = DataPath + '/' + file_name
  if int(curr_line[index_main]) == 1:
      if int(curr_line[index_sub]) == 1:
          destFile = os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, tag_to_use_sub + addTest, add_eval + curr_line[0] + '.' + fileEnder)
          numOf_A_1 += 1
          f_A_1.write(curr_line[0] + '\n')
          copyfile(srcFile, destFile)

      else:
          destFile = os.path.join(path, tag_to_use_main, tag_to_use_sub, tag_to_use_main, 'not_' + tag_to_use_sub + addTest, add_eval + curr_line[0] + '.' + fileEnder)
          numOf_A_2 += 1
          f_A_2.write(curr_line[0] + '\n')
          copyfile(srcFile, destFile)
  else:
      if int(curr_line[index_sub]) == 1:
          destFile = os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, tag_to_use_sub + addTest, add_eval + curr_line[0] + '.' + fileEnder)
          numOf_B_1 += 1
          f_B_1.write(curr_line[0] + '\n')
          copyfile(srcFile, destFile)
      else:
          destFile = os.path.join(path, tag_to_use_main, tag_to_use_sub, 'not_' + tag_to_use_main, 'not_' + tag_to_use_sub + addTest, add_eval + curr_line[0] + '.' + fileEnder)
          numOf_B_2 += 1
          f_B_1.write(curr_line[0] + '\n')
          copyfile(srcFile, destFile)

tot_A = numOf_A_1 + numOf_A_2
tot_B = numOf_B_1 + numOf_B_2

print('Total ' + tag_to_use_main + ': ' + str(tot_A))
f_all_info.write('Total ' + tag_to_use_main + ': ' + str(tot_A) + '\n')

print('Total not_' + tag_to_use_main + ': ' + str(tot_B))
f_all_info.write('Total_not ' + tag_to_use_main + ': ' + str(tot_B) + '\n')

print('Total ' + tag_to_use_main + ' + ' + tag_to_use_sub + ': ' + str(numOf_A_1))
f_all_info.write('Total ' + tag_to_use_main + ' + ' + tag_to_use_sub + ': ' + str(numOf_A_1) + '\n')

print('Total ' + tag_to_use_main + ' + not_' + tag_to_use_sub + ': ' + str(numOf_A_2))
f_all_info.write('Total ' + tag_to_use_main + ' + not_' + tag_to_use_sub + ': ' + str(numOf_A_2) + '\n')

print('Total not_' + tag_to_use_main + ' + ' + tag_to_use_sub + ': ' + str(numOf_B_1))
f_all_info.write('Total not_' + tag_to_use_main + ' + ' + tag_to_use_sub + ': ' + str(numOf_B_1) + '\n')

print('Total not_' + tag_to_use_main + ' + not_' + tag_to_use_sub + ': ' + str(numOf_B_2))
f_all_info.write('Total not_' + tag_to_use_main + ' + not_' + tag_to_use_sub + ': ' + str(numOf_B_2) + '\n')

ratio_A_1 = numOf_A_1 / tot_A
ratio_A_2 = numOf_A_2 / tot_A
ratio_B_1 = numOf_B_1 / tot_B
ratio_B_2 = numOf_B_2 / tot_B

ratio_A_1_str = "{:.3f}".format(ratio_A_1)
ratio_A_2_str = "{:.3f}".format(ratio_A_2)
ratio_B_1_str = "{:.3f}".format(ratio_B_1)
ratio_B_2_str = "{:.3f}".format(ratio_B_2)


f_all_info.write('---------------------------\n')

print(tag_to_use_main + ' : ' + tag_to_use_sub + '/not_' + tag_to_use_sub + ' : ' + ratio_A_1_str + '/' + ratio_A_2_str)
f_all_info.write(tag_to_use_main + ' : ' + tag_to_use_sub + '/not_' + tag_to_use_sub + ' : ' + ratio_A_1_str + '/' + ratio_A_2_str + '\n')

print('not_' + tag_to_use_main + ' : ' + tag_to_use_sub + '/not_' + tag_to_use_sub + ' : ' + ratio_B_1_str + '/' + ratio_B_2_str)
f_all_info.write('not_' + tag_to_use_main + ' : ' + tag_to_use_sub + '/not_' + tag_to_use_sub + ' : ' + ratio_B_1_str + '/' + ratio_B_2_str + '\n')

f.close()
f_eval.close()
f_A_1.close()
f_A_2.close()
f_B_1.close()
f_B_2.close()
f_all_info.close()
