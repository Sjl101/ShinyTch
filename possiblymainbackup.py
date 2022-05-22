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

outputname = ''
outputscore = 0
encountercount = 0

title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
mon = (0, 40, 1920, 1080) 
while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
 
model_dir = pathlib.Path("E:/Models/pokemon_v13_output/saved_model/")
model = tf.saved_model.load(str(model_dir))
 
PATH_TO_LABELS = 'data/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
 
detection_model = model



default_app = firebase_admin.initialize_app()
shinysprite = False
selectedpokemon = ''
selectedsprites = 'assets/sprites/'

selection = (
    '001-bulbasaur', '002-ivysaur', '003-venusaur', '004-charmander', '005-charmeleon', '006-charizard',
    '007-squirtle', '008-wartortle', '009-blastoise', '010-caterpie', '011-metapod',
    '012-butterfree', '013-weedle', '014-kakuna', '015-beedrill', '016-pidgey',
    '017-pidgeotto', '018-pidgeot', '019-rattata', '020-raticate', '021-spearow', '022-fearow',
    '023-ekans', '024-arbok', '025-pikachu', '026-raichu', '027-sandshrew', '028-sandslash',
    '029-nidoran-f', '030-nidorina', '031-nidoqueen', '032-nidoran-m', '033-nidorino', '034-nidoking',
    '035-clefairy', '036-clefable')



gif = False
def main():
    global selectedsprites
    global selectedpokemon
    global fps
    global start_time
    global display_time
    global outputname
    global outputscore  
    global encountercount 
    print(default_app)
    defaultgif = 'assets/default.gif'
    sg.theme("Purple")
    image_viewer_column = [
        [sg.Button("Hunt Settings", size=(20, 1), key='settings', tooltip="Settings for the current hunt")],
        [sg.Text("Selected Pokemon:" + selectedpokemon)],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-pokemon-")],
    ]
    # Define the window layout
    layout = [
        [sg.Text("Live Video", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-"),sg.VSeperator(),sg.Column(image_viewer_column)],
        [sg.Listbox(values=selection, select_mode='single', key='-fac-', size=(30, 6), enable_events=True, tooltip="Select pokemon to hunt")],
        [sg.Button("Exit", size=(10, 1))],
        [sg.Text(size=(8, 2), font=('Helvetica', 20), justification='center', key='text')],
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

        Imagenp=show_inference(detection_model, frame)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        window["text"].update(encountercount)
        
        # print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
        # plt.imshow('object detection', cv2.resize(Imagenp, (800,600)))

        event, values = window.Read(timeout=1) # every 100ms, fire the event sg.TIMEOUT_KEY
        window.find_element("-pokemon-").UpdateAnimation(defaultgif,time_between_frames=100)
        
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            print(outputname)
            encountercount + 1
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
    global shinysprite
    global selectedsprites
    layout = [
        [sg.Text("Hunt Settings", key="hunt")],
        [sg.Text("Sprite Settings:", key="hunt")],
        [sg.Button("Shiny", key="shinyspriteswitch")],
        [sg.Button("Sprite Change", key="source")],
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
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    huntsettingswindow.close()






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
    outputname = output_dict['detection_classes'].astype(np.int64)
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
        line_thickness=5)
    outputname = category_index[output_dict['detection_classes'][np.argmax(output_dict['detection_scores'])]+1]['name']
    return(image_np)



#Now we open the webcam and start detecting objects
import cv2



main()