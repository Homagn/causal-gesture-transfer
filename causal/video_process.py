import cv2
import json
import os
import glob
import random

#label_ids = [40,42,45,47,55,  86,87,93,94,   99,100,101] #tasks for these label ids can be translated into robot task

def vid2frames(fname, savefolder):
    #pass a video filename and extract all its frames into savefolder location
    vidcap = cv2.VideoCapture(fname)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(savefolder+"/%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      #print('Read a new frame: ', success)
      count += 1

def get_vids(tasks, label_files_loc):
    #Data in something-something is arranged in 2 folders one contains only videos with names as id such as '112.webm', '1231.webm' and so on
    #the other folder contains json files train,test,val,label
    #train json file contains a list of dictionaries. The 'template' key of that dictionary describes the task and 'id' key 
    #gives the name of the video in the video folder
    #this function gives the list of all ids for videos of a certain kind of task

    with open(label_files_loc, "r", encoding='utf-8') as f:
        labels = json.load(f)
    vids = []
    for l in labels:
        if tasks[0] == l["template"]:
            vids.append(l["id"])
    #print(vids)
    print("Number of videos obtained for the task ",len(vids))
    return vids

def extract_task_frames(video_files_loc, vids, savefolder):
    path = os.path.join('/causal/', savefolder) 
    try:
        os.mkdir(path)
    except:
        print("Path already exists ")

    for v in vids:
        fname = video_files_loc+v+'.webm'
        path = os.path.join('/causal/'+savefolder+'/', v) 
        try:
            os.mkdir(path)
        except:
            print("Path already exists ")
        sf = savefolder+'/'+v
        print("saving frames at ",sf)
        vid2frames(fname, sf)


def sample_data(vid, extracted_task_images_loc):
    #pass a video id number corresponding to a task and get a datapoint for training
    s = len(extracted_task_images_loc+'/'+vid+'/')
    frame_n = []
    for i in glob.glob(extracted_task_images_loc+'/'+vid+'/*'):
        frame_n.append(int(i[s:i.index('.')])) #get the exact number of the frame

    max_n = max(frame_n)
    f = random.choice(list(range(max_n)))
    print("randomly chosen frame number ",f)
    
    def processed_image_number(f, resized = (128,128)):
        image_name = extracted_task_images_loc+'/'+vid+'/'+repr(f)+'.jpg'
        image = cv2.imread(image_name)
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bw_resize = cv2.resize(image_bw, resized)
        return image_bw_resize
    
    cv2.imshow('chosen_image_init', processed_image_number(0))
    cv2.imshow('chosen_image_current_random', processed_image_number(f))
    cv2.imshow('chosen_image_final', processed_image_number(max_n))
    cv2.waitKey(0)

    completion_ratio = float(f/max_n)
    print("completion ratio of current image ",completion_ratio)




tasks = ["Pulling [something] from left to right"]
label_files_loc = r"something-something-labels/something-something-v2-train.json"
video_files_loc = 'something-something/20bn-something-something-v2/'
extracted_task_images_loc = 'causal_tasks'


vids = get_vids(tasks,label_files_loc)[0:10]
print("First 10 video ids ",vids)


extract_task_frames(video_files_loc,vids, extracted_task_images_loc)

sample_data("9300",extracted_task_images_loc)


