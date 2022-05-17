import PySimpleGUI as sg
import cv2
import numpy as np
import os.path

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
    defaultgif = 'assets/default.gif'
    sg.theme("Purple")
    image_viewer_column = [
        [sg.Text("Selected Pokemon:")],
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
    window.set_icon("assets/icon.png")

    cap = cv2.VideoCapture(0)
    
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        folder = 'assets/'
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

        event, values = window.Read(timeout=1) # every 100ms, fire the event sg.TIMEOUT_KEY
        window.find_element("-pokemon-").UpdateAnimation(defaultgif,time_between_frames=100)

        if event == "-fac-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(folder, values["-fac-"][0] + '.gif')
                window["-TOUT-"].update(filename)
                window["-pokemon-"].update(filename=filename)
                defaultgif = filename
                gif == True
            except:
                pass  
    window.close()

main()