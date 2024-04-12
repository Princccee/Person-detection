import cv2
import os


def detect_faces(image):
    # Load the cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces


def calculate_face_area(faces):
    # Calculate the total area of all detected faces
    total_area = 0
    for (x, y, w, h) in faces:
        total_area += w * h

    # Calculate the average face area
    if len(faces) > 0:
        average_area = total_area / len(faces)
    else:
        average_area = 0

    return average_area


def choose_best_image(images_folder):
    best_image = None
    best_area = 0

    # Iterate through all images in the folder
    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            # Read the image
            image = cv2.imread(image_path)
            # Detect faces in the image
            faces = detect_faces(image)
            # Calculate the average face area
            average_area = calculate_face_area(faces)
            # Update the best image if needed
            if average_area > best_area:
                best_image = image
                best_area = average_area

    return best_image


def main():
    # Folder containing the set of images
    images_folder = 'snapshots'

    # Choose the best image from the set
    best_image = choose_best_image(images_folder)

    # Display or save the best image
    if best_image is not None:
        cv2.imshow('Best Image with Detected Faces', best_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('best_image_with_faces.jpg', best_image)
        print("Best image with detected faces saved as 'best_image_with_faces.jpg'")
    else:
        print("No images found or no faces detected in the provided images.")


if __name__ == "__main__":
    main()
