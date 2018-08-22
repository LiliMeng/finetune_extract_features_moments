import numpy as np


def make_new_list(img_list, new_img_list):
	
	new_file_with_img_num = open(new_img_list, "a")
	lines = [line.strip() for line in open(img_list).readlines()]

	for line in lines:
	    videoname = line.split(',')[0]
	    label = line.split(',')[1]
	    print("videoname: ", videoname)
	    print("label: ", label)
	    new_file_with_img_num.write(videoname+" "+label+" "+str(15)+"\n") 

	new_file_with_img_num.close()

def main():

	train_img_list = "./moments_train_list.txt"
	new_train_img_list = "./new_moments_train_list.txt"

	val_img_list = "./moments_validation_list.txt"
	new_val_img_list = "./new_moments_validation_list.txt"

	make_new_list(train_img_list, new_train_img_list)
	make_new_list(val_img_list, new_val_img_list)

if __name__== "__main__":
  main()
