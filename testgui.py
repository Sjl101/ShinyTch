import PySimpleGUI as sg
import cv2
import numpy as np
import os.path
import firebase_admin
import json

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
        
    ]

    # Create the window and show it without the plot
    window = sg.Window("PokeTch", location=(400, 200), ).Layout(layout)
    window.set_icon("assets/icon2.png")

    cap = cv2.VideoCapture(0)
    
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

        event, values = window.Read(timeout=1) # every 100ms, fire the event sg.TIMEOUT_KEY
        window.find_element("-pokemon-").UpdateAnimation(defaultgif,time_between_frames=100)

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
    
main()