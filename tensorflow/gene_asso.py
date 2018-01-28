with open("associations.txt",'a') as the_file:
	for idx in range(0, 1352):
		base_filename = "_out" + str(idx+1).zfill(6) + ".png"
		lineStr = str(idx) + " color_images/color" + base_filename + " " + str(idx) + " depth_images/depth" + base_filename
		the_file.write(lineStr + '\n')
