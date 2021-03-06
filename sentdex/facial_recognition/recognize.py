import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
    # Default tolerance for facial recognition is 0.6
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
 # CNN = Convolutional neural network
 # Alternative is HOG and may work better on cpu only
MODEL = "hog"   # cnn or hog

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
            # [0] to only encode the first face it finds (for the pictures of the person in the known images)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
        print(name)

    # Now we have the known faces, we want to go through all the unknown faces and check to see if we have any matches
print()
print()
print("processing unknown faces")
filesWithMatches = []
loading = "+"
for filename in os.listdir(UNKNOWN_FACES_DIR):

    print(loading)
    loading += "+"
    if loading == "+++++++++++++++":
        loading = "+"

    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
        # There may be multiple faces in the unknown photos, we need to detect all faces before trying to recognise them
    locations = face_recognition.face_locations(image, model=MODEL)
        # We need to record the locations of the faces too so we can label them later
    encodings = face_recognition.face_encodings(image, locations)
        # Convert the image so it can be used with open cv
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check for matches
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            filesWithMatches.append(filename)
            match = known_names[results.index(True)]
            print(f"\tMatch found: {match}")
            print(f"\tIn: {filename}")


                # Show image with a box around the identified face
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
                
                # Make the box around the face
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Make a text box for the name
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, 
                        match, 
                        (face_location[3]+10, face_location[2]+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (200, 200, 200), 
                        FONT_THICKNESS)

    # cv2.imshow(filename, image)
    cv2.waitKey(10000)
    # cv2.destroyWindow(filename)   <-- doesn't work on Ubuntu

print(f"Files with matches: {filesWithMatches}")
