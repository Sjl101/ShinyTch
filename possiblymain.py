from enum import auto
import PySimpleGUI as sg
import cv2
import numpy as np
import os.path
import firebase_admin
import json
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageOps 
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from grabscreen import grab_screen
import time



#sets veriables i need later on
left, right, top, bottom = 200, 600, 300, 730
width, height = 1920, 1080
coords = []
encountertime = 20
outputname = ''
score = 0.0
outputscore = 0
encountercount = 0
selectedpokemon = ''
selectedsprites = 'assets/'
encountertimer = 0.0
model_dir = pathlib.Path("E:/Models/pokemon_v14_output/saved_model/")
PATH_TO_LABELS = 'data/label_map.pbtxt'
shinysprite = False
selection = ('001-bulbasaur', '002-ivysaur', '003-venusaur', '004-charmander', '005-charmeleon', '006-charizard','007-squirtle', '008-wartortle', '009-blastoise')
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
encount_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
mon = (0, 40, 1920, 1080) 
while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')

model = tf.saved_model.load(str(model_dir)) 
detection_model = model
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

default_app = firebase_admin.initialize_app()
#----------------------------------------------------------------
#functions

#main function
def main():
    global selectedsprites
    global selectedpokemon
    global fps
    global start_time
    global display_time
    global outputname
    global outputscore  
    global encountercount, encountertimer
    global width, height
    global left, right, top, bottom
    print(default_app)
    defaultgif = 'assets/default.gif'
    sg.theme("Purple")
    image_viewer_column = [
        [sg.Button("Hunt Settings", size=(20, 1), key='settings', tooltip="Settings for the current hunt")],
        [sg.Text("Selected Pokemon:" + selectedpokemon)],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-pokemon-")],
        [sg.Text(size=(8, 2), font=('Helvetica', 20), justification='center', key='text')],
    ]
    # Define the window layout
    layout = [
        [sg.Text("Live Video", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-"),sg.VSeperator(),sg.Column(image_viewer_column)],
        [sg.Listbox(values=selection, select_mode='single', key='-fac-', size=(30, 6), enable_events=True, tooltip="Select pokemon to hunt"),sg.Text(size=(8, 2), font=('Helvetica', 20), justification='center', key='enctime')],
        [sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("PokeTch", location=(400, 200), ).Layout(layout)
    window.set_icon("assets/icon2.png")

    cap = cv2.VideoCapture(0)
    
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        

        frame = grab_screen(region=mon)
        count =0
        Imagenp=show_inference(detection_model, frame)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        #cv2.imshow('object detection', cv2.resize(frame, (1280, 720)))
        


        encountercount = encounter(outputname, encountercount, selectedpokemon, True, frame)
        window["text"].update(encountercount)
        window["enctime"].update(encountertimer)
        # print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
        # plt.imshow('object detection', cv2.resize(Imagenp, (800,600)))

        event, values = window.Read(timeout=1) # every 100ms, fire the event sg.TIMEOUT_KEY
        window.find_element("-pokemon-").UpdateAnimation(defaultgif,time_between_frames=100)
        
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            print(outputname)
            print(encountercount)
            print(f"time left:{encountertimer}")
            fps = 0
            start_time = time.time()

        if event == "-fac-":  # A file was chosen from the listbox
            try:
                if shinysprite == True:
                    selectedpokemon = values["-fac-"][0]
                    print(selectedpokemon)
                    filename = os.path.join(selectedsprites, values["-fac-"][0] + '-s.gif')
                elif shinysprite == False:
                    selectedpokemon = values["-fac-"][0]
                    print(selectedpokemon)
                    filename = os.path.join(selectedsprites, values["-fac-"][0] + '.gif')

                window["-TOUT-"].update(filename)
                window["-pokemon-"].update(filename=filename)
                defaultgif = filename
            except:
                pass  
        if event == 'settings':
            settings_window()
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(selectedpokemon, f, ensure_ascii=False, indent=4) 
        
    window.close()

#code for the hunt settings window
def settings_window():
    #gets the shinysprite bool 
    global start_time
    global shinysprite
    global selectedsprites
    global encountertime
    layout = [
        [sg.Text("Hunt Settings", key="hunt")],
        [sg.Text("Sprite Settings:", key="hunt")],
        [sg.Button("Shiny", key="shinyspriteswitch")],
        [sg.Button("Sprite Change", key="source")],
        [sg.Text(f"Time Between Encounters: (current = {encountertime}")],
        [sg.Input(size=(200,200))],
    ]
    huntsettingswindow = sg.Window("Hunt Settings", layout, modal=True)
    choice = None
    while True:
        event, values = huntsettingswindow.read(timeout=20)
        if event == 'shinyspriteswitch':
            if shinysprite == False:
                shinysprite = True
                print(shinysprite)
            elif shinysprite == True:
                shinysprite = False
                print(shinysprite)
        if event == 'source':
            if selectedsprites == 'assets/sprites/':
                selectedsprites = 'assets/'
                print(selectedsprites)
            elif selectedsprites == 'assets/':
                selectedsprites = 'assets/sprites/'
                print(selectedsprites)
        try:
            if values[0] == '':
                encountertime = 20
            else:
                try:
                    encountertime = float(values[0])
                except TypeError as s:
                    print(s)
                    print("Not a number")
        except TypeError as e:
            print(e)
            print("handled successfully")
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    huntsettingswindow.close()

def encounter(pokemon, encounterscount, selectedpokemonfunc, switch, frame):
    global encount_time
    global encountertime
    global coords
    global width, height
    global left, right, top, bottom
    global encountertimer
    result = encounterscount
    TIMEENC = time.time() - encount_time
    #print(f'time left: {TIMEENC}')
    encountertimer = TIMEENC
    encountertimer = int(encountertimer) 
    if (TIMEENC) >= encountertime:
        if switch == True:
            if pokemon == selectedpokemonfunc:
                result = encounterscount + 1
        encount_time = time.time()
        print(coords)
        #sets the coords for the detected pokemon
        xmin = coords[0]
        ymin = coords[1]
        xmax = coords[2]
        ymax = coords[3]
        #makes them bigger to be usable with the resulution 
        (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
        im = Image.fromarray(frame)
        #gets the coords ready to crop
        a,b,c,d = int(left) , int(right) , int(top) ,int(bottom)
        h = d - c
        w = b - a 
        #shows coords
        print((c,a,d,b))
        #crops
        im2 = im.crop((c,a,d,b))  
        hw = (w, h)
           
        im3 = im2.resize(hw)  
        im3.save("temp.png")
        readimg = cv2.imread('temp.png')
        #converts the image to grayscale to get rid of the blackboxes
        gray, thresh, rgbcrop = remove_black_box(readimg)
        
        cv2.imwrite('sofwinres.png',rgbcrop)
        #gets the center coordinates of the detected pokemon
        cX, cY = find_center(thresh,readimg, w , h)
        gray2, thresh2, rgbcrop2 = remove_black_box(readimg)
        cv2.imwrite('centercrop.png',rgbcrop2)
    return result

def find_center(thresh, img, w, h):
    red = [0,0,255]
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite("result.png", rgb)
    return cX, cY
        
def remove_black_box(readimg):
    gray = cv2.cvtColor(readimg,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = readimg[y:y+h,x:x+w]
    #converts the to correct rgb values
    rgbcrop = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    return gray, thresh, rgbcrop

def run_inference_for_single_image(model, image):
    global outputname
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
 
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
 
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
 
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
     
    return output_dict

def show_inference(model, frame):
    global outputname
    global outputscore   
    global coords
    global score
  #take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
  # Actual detection.
     
    output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=0.5,
        line_thickness=5)
    score = output_dict['detection_scores'][np.argmax(output_dict['detection_scores'])]
    coords = output_dict['detection_boxes'][np.argmax(output_dict['detection_scores'])]
    outputname = category_index[output_dict['detection_classes'][np.argmax(output_dict['detection_scores'])]]['name']
    return(image_np)
main()