import cv2
import os
import face_recognition

recognition_data = {'name':[],'coords':[]}

def load_images():
    """
    This section loads all existing pics from images folder in memory (RAM).

    It is necessary because it enhances detection and recognition speed.
    """
    image_folder = 'images'
    for files in os.listdir(image_folder):

        image_path = os.path.join(image_folder,files)

        try:
            new_encs = face_recognition.face_encodings(cv2.imread(image_path))

            for encs in new_encs:
                #Storing names and coordinates in dictionary
                recognition_data["name"].append(files.split('.')[0])
                recognition_data["coords"].append(encs)

                print('Loading '+files+'....')
        except Exception as e:
            print('Error loading '+files+'....\n',e)

def recognize_person(frame):
    """
    This function checks if a person from
    given frame exists in our known people
    list.
    """

    try:
        face_encs = face_recognition.face_encodings(frame)
    except Exception as e:
        print('Unable to detect:', e)  # Debugging statement
        return None

    if face_encs != []:
        coords = face_encs[0]
            
        for i, coord in enumerate(recognition_data['coords']):
            res = face_recognition.compare_faces([coord], coords, tolerance=0.5)
            if res[0]:
                return recognition_data['name'][i]

    return 'Unknown'


load_images()
source = input('Enter video source (0 for webcam): ')

if source.isnumeric():
    source = int(source)

cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()

    if not ret:
        print('Video source expired..')

    person = recognize_person(frame)

    cv2.putText(frame, person, (10, 50), 2, 2, (0,255,0), 1)
    cv2.imshow('Output',frame)
    if cv2.waitKey(1) == ord('q'):
        break
