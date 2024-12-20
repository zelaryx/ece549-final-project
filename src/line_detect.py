import cv2
import numpy as np
import os

def greyfill(img):
    #img = cv2.imread('output_clean_p1.jpg', cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    img = img/255

    #img = img[0:800, :]
    threshold = 0.5
    #create binarized image to enhance lines
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #print(blurred_img[i, j])
            if(img[i, j] < threshold):
                img[i, j] = 0
            else:
                img[i, j] = 1

    #apply gaussian blur to binarized image
    #rectangular kernel where height < height of vertical gap between text lines and width larger
    #img_avg = cv2.GaussianBlur(img, (101, 3), 0, 0)

    #img = np.empty((img_avg.shape[0], img_avg.shape[1]))
    #threshold_2 = 0.9
    #binarize img_avg to get initial estimate of text lines
    #for i in range(0, img_avg.shape[0]):
    #    for j in range(0, img_avg.shape[1]):
    #        #print(blurred_img[i, j])
    #        if(img_avg[i, j] < threshold_2):
    #            img[i, j] = 0
    #        else:
    #            img[i, j] = 1

    #start filling operation:
    fv = 0.7
    #grayfill_img = np.zeros((img.shape[0], img.shape[1]))
    s = 0
    e = 0
    u = 0
    r = 8
    r_width = img.shape[1]//r
    counter = 0
    for m1 in range(0, img.shape[0]):
        counter = 0
        while(counter < r):
            step_counter = 0
            #left fill
            for m2 in range(counter*r_width, (counter+1)*r_width):
                if m2 < img.shape[1]:
                    if img[m1, m2] == 0:
                        if(step_counter < 0.2*r_width and counter != 0):
                            img[m1, counter*r_width: counter*r_width+step_counter] = 1
                        break
                    else:
                        step_counter += 1
                        img[m1, m2] = fv
            step_counter = 0
            for m2 in range((counter+1)*r_width, counter*r_width, -1):
                if (m2 < img.shape[1]):
                    if img[m1, m2] == 0:
                        if(step_counter < 0.2*r_width and counter != r-1):
                            img[m1, (counter+1)*r_width-step_counter:(counter+1)*r_width] = 1
                        break
                    else:
                        step_counter += 1
                        img[m1, m2] = fv
            counter += 1

    #reprocess img to remove noise
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] != fv and img[i+1, j] == fv  and img[i-1, j] == fv and img[i, j+1] == fv and img[i, j-1] == fv:
                img[i, j] = fv
    #cv2.imshow("fill_igm", img)
    #cv2.waitKey(0)
    return img


def detect_line_start(img):
    #read image in as greyscale
    #img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = img/255

    #img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    threshold = 0.5
    #create binarized image to enhance lines
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #print(blurred_img[i, j])
            if(img[i, j] < threshold):
                img[i, j] = 0
            else:
                img[i, j] = 1

    #apply gaussian blur to binarized image
    #rectangular kernel where height < height of vertical gap between text lines and width larger
    img_avg = cv2.GaussianBlur(img, (101, 3), 0, 0)

    img_initial = np.empty((img_avg.shape[0], img_avg.shape[1]))
    threshold_2 = 0.9
    #binarize img_avg to get initial estimate of text lines
    for i in range(0, img_avg.shape[0]):
        for j in range(0, img_avg.shape[1]):
            #print(blurred_img[i, j])
            if(img_avg[i, j] < threshold_2):
                img_initial[i, j] = 0
            else:
                img_initial[i, j] = 1

    #inverts initial_estimate for cc analysis
    for i in range(0, img_avg.shape[0]):
        for j in range(0, img_avg.shape[1]):
            #print(blurred_img[i, j])
            if(img_initial[i, j] == 0):
                img_initial[i, j] = 1
            else:
                img_initial[i, j] = 0

    #img_initial = img_initial[0:800, :]
    img_initial = img_initial.astype(np.uint8)

    analysis = cv2.connectedComponentsWithStats(img_initial, 4, cv2.CV_32S) 
    (num_labels, label_ids, values, centroid) = analysis
   #print(f"heights: {heights}")
    filtered_heights = np.empty((0))
    for i in range(1, values.shape[0]):
        if values[i, 4] > 500: #checks area
            filtered_heights = np.append(filtered_heights, [values[i, 3]])

    #print(f"filtered_heights : {filtered_heights}")
    cc_med = int(np.median(filtered_heights))
    cc_avg = int(np.average(filtered_heights))
    #print(f"median height: {cc_med}")
    #print(f"average height: {cc_avg}")
    #print(f"cc heights: {heights}")
    num_comps = 0
    output = np.zeros(img_initial.shape, dtype="uint8")
    for i in range(1, num_labels): 
        area = values[i, cv2.CC_STAT_AREA]   
    
        if (area > 500): 
            num_comps += 1
            componentMask = (label_ids == i).astype("uint8") * 255
            
            # Creating the Final output mask 
            output = cv2.bitwise_or(output, componentMask) 

    #cv2.imshow('temp1', img)
    coordinates = np.empty([0])
    counter = 0
    num_lines_found = 0
    for h in range(0, output.shape[0]-(cc_avg + 1)):
        if(counter > cc_avg):
            total_pixels = cc_med*img_initial.shape[1]
            num_pixels = np.count_nonzero(output[h:h+cc_med, :] == 255)
            if(num_pixels > total_pixels*0.075):
                coordinates = np.append(coordinates, [h + cc_med/2])
                num_lines_found += 1
                counter = 0
        counter += 1

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            #print(blurred_img[i, j])
            if(output[i, j] == 0):
                output[i, j] = 255
            else:
                output[i, j] = 0
    return coordinates.astype(int)

def line_separator(greyfill_img, coordinates): 
    line_separator = np.empty((coordinates.shape[0]), dtype=object)
    count = 0
    for c in coordinates:
        #print(f"coordinate: {c}")
        curr_w = 0
        curr_h = c
        current_path = []
        current_path.append([c, 0])
        flag = 'move_right_and_up'
        up = False
        down = False
        failed_up = False
        failed_down = False
        #move right
        while curr_w < greyfill_img.shape[1]-10 and curr_h < greyfill_img.shape[0]-1:
            #Eprint(f"current_h, current_w: {curr_h} {curr_w}")
            #print(f"flag: {flag}")
            if flag == 'move_right_and_up' and failed_up == False:
                #first always try to move right
                if(greyfill_img[curr_h, curr_w+1]) == 0.7 and up == False: 
                    failed_down = False
                    curr_w += 1
                    current_path.append([curr_h, curr_w])
                else:
                    up = True
                #then try to move down
                #first check immediate up, then diagonal, then backtrack
                    if greyfill_img[curr_h-1, curr_w] == 0.7:
                        up = False
                        curr_h -= 1
                        current_path.append([curr_h, curr_w])
                    elif greyfill_img[curr_h-1, curr_w-1] == 0.7:
                        up = False
                        curr_h -= 1
                        curr_w -= 1
                        current_path.append([curr_h, curr_w])
                    elif greyfill_img[curr_h-1, curr_w+1] == 0 and greyfill_img[curr_h-1, curr_w] == 1:
                        failed_up = True
                        flag = 'move_right_and_down'
                    elif greyfill_img[curr_h-1, curr_w] == 0 and greyfill_img[curr_h-1, curr_w-1] == 1:
                        failed_up = True
                        flag = 'move_right_and_down'
                    else:
                        curr_w -= 1
                        current_path.append([curr_h, curr_w])
            if flag == 'move_right_and_down' and failed_down == False:
                #first always try to move right
                #print(down)
                if(greyfill_img[curr_h, curr_w+1]) == 0.7 and down == False: 
                    failed_up = False
                    curr_w += 1
                    current_path.append([curr_h, curr_w])
                else:
                #first check immediate down, then diagonal, then backtrack
                    down = True
                    if greyfill_img[curr_h+1, curr_w] == 0.7:
                        down = False
                        curr_h += 1
                        current_path.append([curr_h, curr_w])
                    elif greyfill_img[curr_h+1, curr_w-1] == 0.7:
                        down = False
                        curr_h += 1
                        curr_w -= 1
                        current_path.append([curr_h, curr_w])
                    elif greyfill_img[curr_h+1, curr_w+1] == 0 and greyfill_img[curr_h+1, curr_w] == 1:
                        failed_down = True
                        flag = 'move_right_and_up'
                    elif greyfill_img[curr_h+1, curr_w] == 0 and greyfill_img[curr_h+1, curr_w-1] == 1:
                        failed_down = True
                        flag = 'move_right_and_up'
                    else:
                        curr_w -= 1
                        current_path.append([curr_h, curr_w])

            if failed_up and failed_down:
                curr_w += 1
                current_path.append([curr_h, curr_w])
                while(greyfill_img[curr_h, curr_w] != 0.7):
                    curr_w += 1
                    current_path.append([curr_h, curr_w])
                flag = 'move_right_and_up'
                up = False
                down = False
                failed_up = False
                failed_down = False
        line_separator[count] = np.array(current_path)
        count += 1
    return line_separator

def output_lines(lines, img):
    old_start = 0

    line_count = 0
    keep_lines = np.empty((lines.shape[0]), dtype=object)
    keep_heights = []

    for i in range(0, lines.shape[0]):
        path = lines[i]
        #find min/max heights 
        min_coord = np.min(path, axis=0)
        max_coord = np.max(path, axis=0)
        new_start = max_coord[0]

        height = max_coord[0] - min_coord[0]
        width = max_coord[1] - min_coord[1]

        line_img = np.ones((new_start - old_start, width))

        for h in range(0, line_img.shape[0]):
            for c in range(0, line_img.shape[1]):
                line_img[h, c] = img[old_start+h, c]

        line_start_height = old_start
        old_start = min_coord[0]

        if(line_img.shape[0]>0):
            tot_pixels = 0
            black_pixels = 0
            for i in range(0, line_img.shape[0]):
                for j in range(0, line_img.shape[1]):
                    tot_pixels += 1
                    if line_img[i, j] != 1:
                        black_pixels += 1

            name = "text_line" + str(line_count) + " (PRESS ENTER TO KEEP, SPACE TO DISCARD)"

            cv2.imshow(name, line_img/255)
            key = cv2.waitKey(0)

            if key == 13:
                keep_lines[line_count] = line_img
                keep_heights.append(line_start_height)
                line_count += 1
            elif key == 32:
                pass

            cv2.destroyAllWindows()

            # cv2.imwrite(name, line_img)
    return keep_lines, keep_heights

def line_detect(img):
    img = img[50:img.shape[0]-50, 50:img.shape[1]-50]
    # cv2.imshow("cropped_img", img)
    # cv2.waitKey(0)
    #img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    greyfill_img = greyfill(img)
    #cv2.imshow("greyfill_img", img)
    coordinates = detect_line_start(img)
    #print(coordinates)
    lines = line_separator(greyfill_img, coordinates)

    lines, heights = output_lines(lines, img)
    return [lines.shape, len(heights), heights]
