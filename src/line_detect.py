import cv2
import numpy as np

def greyfill():
    img = cv2.imread('line_sample.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    img = img/255

    img = img[0:800, :]
    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    threshold = 0.5
    #create binarized image to enhance lines
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #print(blurred_img[i, j])
            if(img[i, j] < threshold):
                img[i, j] = 0
            else:
                img[i, j] = 1

    print(img)
    cv2.imshow("binary img", img)
    #start filling operation:
    fv = 0.7
    grayfill_img = np.zeros((img.shape[0], img.shape[1]))
    s = 0
    e = 0
    u = 0
    r = 8
    r_width = img.shape[1]//r
    print(r_width)
    print(img.shape)
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
    img[100, 100] = 0
    #for m1 in range(0, img.shape[0]):
    #    counter = 0
    #    while(counter < r):
            #print(counter)
    #        for m2 in range(counter*r_width, (counter+1)*r_width):
    #            img[m1, m2] = 0.2
    #        counter += 1
    print(img.shape)
    #cv2.imshow("filled img", img)
    #cv2.waitKey(0)
    return img


def detect_line_start():
    #read image in as greyscale
    img = cv2.imread('line_sample.jpg', cv2.IMREAD_GRAYSCALE)
    img = img/255

    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

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

    for i in range(0, img_avg.shape[0]):
        for j in range(0, img_avg.shape[1]):
            #print(blurred_img[i, j])
            if(img_initial[i, j] == 0):
                img_initial[i, j] = 1
            else:
                img_initial[i, j] = 0

    img_initial = img_initial[0:800, :]
    img_initial = img_initial.astype(np.uint8)

    analysis = cv2.connectedComponentsWithStats(img_initial, 4, cv2.CV_32S) 
    (num_labels, label_ids, values, centroid) = analysis
    print(f"Total labels: {num_labels}") 
    print(f"label id: {label_ids.shape}")
    print(f"CC values: {values}")

    heights = values[1:, 3]
    cc_med = int(np.median(heights.ravel()))
    cc_avg = int(np.average(heights.ravel()))
    print(f"median height: {cc_med}")
    print(f"average height: {cc_avg}")
    print(f"cc heights: {heights}")
    num_comps = 0
    output = np.zeros(img_initial.shape, dtype="uint8")
    for i in range(1, num_labels): 
        area = values[i, cv2.CC_STAT_AREA]   
    
        if (area > 500): 
            num_comps += 1
            # Labels stores all the IDs of the components on the each pixel 
            # It has the same dimension as the threshold 
            # So we'll check the component 
            # then convert it to 255 value to mark it white 
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
            if(num_pixels > total_pixels*0.05):
                coordinates = np.append(coordinates, [h + cc_med/2])
                print("found line")
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

    for i in range(0, coordinates.shape[0]):
        #print(coordinates[i])
        cv2.circle(output, (0, int(coordinates[i])), 5, (0,0,255))
    print(f"number of lines detected: {num_lines_found}")
    print(f"reamining components: {num_comps}")
    print(coordinates)
    #cv2.circle(output, (300, 300), 5, (0,0,255))
    #cv2.imshow('img_avg', img_avg)
    #cv2.imshow('output', output)
    #cv2.imshow('img_initial', 255*img_initial)
    #cv2.waitKey(0)
    return coordinates.astype(int)





def line_separator(greyfill_img, coordinates): 
    for c in coordinates:
        print(f"coordinate: {c}")
        curr_w = 0
        curr_h = c
        current_path = np.empty((0))
        current_path = np.append(current_path, [c, 0], axis=0)
        flag = 'move_right_and_up'
        move_up_fail = False
        check_point_w = 0
        up = False
        down = False
        failed_up = False
        failed_down = False
        #move right
        while curr_w < greyfill_img.shape[1]-2:
            #print(f"current_h, current_w: {curr_h} {curr_w}")
            #print(f"flag: {flag}")
            if flag == 'move_right_and_up' and failed_up == False:
                #first always try to move right
                if(greyfill_img[curr_h, curr_w+1]) == 0.7 and up == False: 
                    failed_down = False
                    curr_w += 1
                    current_path = np.append(current_path, [curr_h, curr_w], axis=0)
                else:
                    up = True
                #then try to move down
                #first check immediate up, then diagonal, then backtrack
                    if greyfill_img[curr_h-1, curr_w] == 0.7:
                        up = False
                        curr_h -= 1
                        current_path = np.append(current_path, [curr_h, curr_w], axis=0)
                    elif greyfill_img[curr_h-1, curr_w-1] == 0.7:
                        up = False
                        curr_h -= 1
                        curr_w -= 1
                        current_path = np.append(current_path, [curr_h, curr_w], axis=0)
                    else:
                        curr_w -= 1
                if greyfill_img[curr_h-1, curr_w+1] == 0 and greyfill_img[curr_h-1, curr_w] != 0.7: #detected blocked region
                        failed_up = True
                        #input("Press Enter to continue...")
                        flag = 'move_right_and_down'
            if flag == 'move_right_and_down' and failed_down == False:
                #first always try to move right
                if(greyfill_img[curr_h, curr_w+1]) == 0.7 and down == False: 
                    failed_up = False
                    curr_w += 1
                    current_path = np.append(current_path, [curr_h, curr_w], axis=0)
                else:
                #first check immediate down, then diagonal, then backtrack
                    down = True
                    if greyfill_img[curr_h+1, curr_w] == 0.7:
                        down = False
                        curr_h += 1
                        current_path = np.append(current_path, [curr_h, curr_w], axis=0)
                    elif greyfill_img[curr_h+1, curr_w-1] == 0.7:
                        down = False
                        curr_h += 1
                        curr_w -= 1
                        current_path = np.append(current_path, [curr_h, curr_w], axis=0)
                    elif greyfill_img[curr_h+1, curr_w+1] == 0 and greyfill_img[curr_h+1, curr_w] == 1:
                        failed_down = True
                        flag = 'move_right_and_up'
                        print(f"current flag: {flag}")
                        print(f"current_h, current_w: {curr_h} {curr_w}")
                        #print(f"current move up check: {move_up_fail}")
                        input("Press Enter to continue...")
                    elif greyfill_img[curr_h+1, curr_w] == 0 and greyfill_img[curr_h+1, curr_w-1] == 1:
                        failed_down = True
                        flag = 'move_right_and_up'
                        print(f"current flag: {flag}")
                        print(f"current_h, current_w: {curr_h} {curr_w}")
                        #print(f"current move up check: {move_up_fail}")
                        input("Press Enter to continue...")
                    else:
                        curr_w -= 1

                #if greyfill_img[curr_h+1, curr_w+1] == 0 and greyfill_img[curr_h+1, curr_w] != 0.7: #detected blocked region
                #        print(f"current flag: {flag}")
                #       print(f"current_h, current_w: {curr_h} {curr_w}")
                #        #print(f"current move up check: {move_up_fail}")
                #        input("Press Enter to continue...")
                #        failed_down = True
                #        flag = 'move_right_and_up'
            if failed_up and failed_down:
                print(f"current flag: {flag}")
                print(f"current_h, current_w: {curr_h} {curr_w}")
                #print(f"current move up check: {move_up_fail}")
                input("Press Enter to continue...")
                pass


                
greyfill_img = greyfill()
coordinates = detect_line_start()
line_separator(greyfill_img, coordinates)

for i in range(0, coordinates.shape[0]):
    #print(coordinates[i])
    cv2.circle(greyfill_img, (0, coordinates[i]), 5, (0,0,255))

ph = 67
pw = 125
greyfill_img[ph, pw] = 0.3
print(greyfill_img[ph-7:ph+7, pw-7:pw+7])
#cv2.circle(greyfill_img, (623, 25), 2, (0, 0,255))
cv2.imshow("part", greyfill_img[ph-10:ph+10, pw-10:pw+10])
cv2.imshow("greyfill_img", greyfill_img)
cv2.waitKey(0)