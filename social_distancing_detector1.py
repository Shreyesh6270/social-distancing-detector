import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt



def __generate_partial_image (picture, partial_image, position):
    if not isinstance (position, tuple):
        raise Exception ("position must be a tuple representing x, y coordinates")
    
    image_height, image_width = partial_image.shape [: 2]
    x, y = position
    picture [x: x + image_height, y: y + image_width] = partial_image

def __generate_text (image, text, target_size, font_scale, color, thickness):
    
    cv2.putText (
        image,
        text,
        target_size,
        fontFace = cv2.FONT_HERSHEY_TRIPLEX ,
        fontScale = font_scale,
        color = color,
        thickness = thickness
    )

def __generate_logo (path_image, target_size = (280,100)):
    
    img_logo = cv2.cvtColor (cv2.imread (path_image), cv2.COLOR_BGR2RGB)
    img_logo = cv2.resize (cv2.imread (path_image), target_size)
    return img_logo

def generate_bird_eye_view (good, bad):
    
    green = (20, 255, 57)
    red = (0, 0, 247)
   
    target_size = (600, 1000)

    # Background size
    background = np.zeros ((3000, 4500, 3), dtype = np.uint8)

    # Points that respect the distance
    for point in good:
        cv2.circle(background, tuple (point), 100, green, 3)
        cv2.circle (background, tuple (point), 25, green, -1)
    
    # Points that don't respect the distance
    for point in bad:
        cv2.circle (background, tuple (point), 100, red, 3)
        cv2.circle (background, tuple (point), 25, red, -1)


    # ROI of bird eye view
    cut_posx_min, cut_posx_max = (2000, 3400)
    cut_posy_min, cut_posy_max = (200, 2800)

    bird_eye_view = background [cut_posy_min: cut_posy_max,
                                cut_posx_min: cut_posx_max,
                                :]

    # Bird Eye View resize
    bird_eye_view_resize = cv2.resize (bird_eye_view, target_size)
    return bird_eye_view_resize

def generate_picture ():
   
    text_color = (112, 25, 25)
    target_size = (1250, 2600, 3)
    background = np.ones (target_size, dtype = np.uint8) * 150
    background [0: 120 ,:] = 255
    background [1200:,:] = 255

    # Generate Logo
    path_logo = '/home/shreyesh/project/social.jpg'
    img_logo = __generate_logo (path_logo)
    __generate_partial_image (background, img_logo, position = (10, 25))

    # Generate Title Original
    __generate_text (image = background,
                text = "Social Distancing Detector",
                target_size = (400, 90),
                font_scale = 2,
                color = text_color,
                thickness = 4)

    # Generate Title Bird Eye View
    __generate_text (image = background,
                text = "Bird's Eye View",
                target_size = (1960, 90),
                font_scale = 2,
                color = text_color,
                thickness = 4)


    picture = cv2.copyMakeBorder (background, 2,2,2,2, cv2.BORDER_CONSTANT)
    return picture
def generate_content_view (picture, image, bird_eye_view):
   
    content = picture.copy ()

    # Orginal View
    __generate_partial_image (content, image, position = (120, 0))

    # Bird Eye View
    __generate_partial_image (content, bird_eye_view, position = (160, 1960))

    return content







def __matrix_bird_eye_view ():
 
    return np.array ([[1.14199333e+00, 6.94076400e+00, 8.88203441e+02],
       [-5.13279159e-01, 7.26783411e+00, 1.02467130e+03],
       [9.79674124e-07, 1.99580075e-03, 1.00000000e+00]])

def __map_points_to_bird_eye_view (points):
  
    if not isinstance (points, list):
        raise Exception ("poinst must be a list of type [[x1, y1], [x2, y2], ...]")
    
    matrix_transformation = __matrix_bird_eye_view ()
    new_points = np.array ([points], dtype = np.float32)
    
    return cv2.perspectiveTransform (new_points, matrix_transformation)
    
def eucledian_distance (point1, point2):
  
    x1, y1 = point1
    x2, y2 = point2
    return sqrt ((x1-x2) ** 2 + (y1-y2) ** 2)



def create_model(config, weights):
    model = cv2.dnn.readNetFromDarknet (config, weights)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU
    
    model.setPreferableBackend (backend)
    model.setPreferableTarget (target)
    return model
def get_output_layers(model):
    layer_names = model.getLayerNames ()
    output_layers = [layer_names [i [0] -1] for i in model.getUnconnectedOutLayers ()]
    return output_layers

def blob_from_image(image, target_size):

    if not isinstance (target_size, tuple): 
        raise Exception ("target_size must be a tuple (width, height)")
    
    blob = cv2.dnn.blobFromImage (image, 
                                 1/255.,
                                 target_size,
                                 [0,0,0],
                                 1,
                                 crop = False)
    
    return blob

def predict(blob, model, outputlayers):
    model.setInput (blob)
    outputs = model.forward (output_layers)

    return outputs
def non_maximum_suppression(image, outputs, confidence_threshold = 0.6, nms_threshold = 0.4):
    
    class_ids = []
    confidences = []
    boxes = []

    img_height, img_width = image.shape [: 2]
    
    #detecting bounding boxing
    for output in outputs:
        for detection in output:
            scores = detection [5:]
            class_id = np.argmax (scores)
            confidence = scores [class_id]
            if confidence> confidence_threshold:
                cx = int (detection [0] * img_width)
                cy = int (detection [1] * img_height)
                width = int (detection [2] * img_width)
                height = int (detection [3] * img_height)
                left = int (cx - width / 2)
                top = int (cy - height / 2)
                class_ids.append (class_id)
                confidences.append (float (confidence))
                boxes.append ([left, top, width, height])
    
    nms_indices = cv2.dnn.NMSBoxes (boxes, confidences, confidence_threshold, nms_threshold)
    
    return boxes, nms_indices, class_ids
def get_domain_boxes(classes, class_ids, nms_indices, boxes, domain_class):
    
    domain_boxes = []
    for index in nms_indices:
        idx = index [0]
        class_name = classes [class_ids [idx]]
        if class_name == domain_class:
            box = boxes [idx]
            left = box [0]
            top = box [1]
            width = box [2]
            height = box [3]
            cx = left + int (width / 2)
            cy = top + int (height / 2)
            domain_boxes.append ((left, top, width, height, cx, cy))
    
    return domain_boxes

def people_distances_bird_eye_view(boxes, distance_allowed):
    
    people_bad_distances = []
    people_good_distances = []
    # We take the values ​​center, bottom
    result = __map_points_to_bird_eye_view ([[box [4], box [1] + box [3]] for box in boxes]) [0]
    # We create new bounding boxes with mapped values ​​of bird eye view (8 elements per item)
    # left, top, width, height, cx, cy, bev_cy, bev_cy
    new_boxes = [box + tuple (result) for box, result in zip (boxes, result)]

    for i in range (0, len (new_boxes) -1):
        for j in range (i + 1, len (new_boxes)):
            cxi, cyi = new_boxes [i] [6:]
            cxj, cyj = new_boxes [j] [6:]
            distance = eucledian_distance ([cxi, cyi], [cxj, cyj])
            if distance <distance_allowed:
                people_bad_distances.append (new_boxes [i])
                people_bad_distances.append (new_boxes [j])

    people_good_distances = list (set (new_boxes) - set (people_bad_distances))
    people_bad_distances = list (set (people_bad_distances))
    
    return (people_good_distances, people_bad_distances)


def draw_new_image_with_boxes (image, people_good_distances, people_bad_distances, distance_allowed, draw_lines = False):
  
    green = (20, 255, 57)
    red = (0, 0, 247)
    new_image = image.copy ()
    
    for person in people_bad_distances:
        left, top, width, height = person [: 4]
        #cv2.rectangle (new_image, (left, top), (left + width, top + height), red, 2)
        #cv2.circle(new_image,(left + 50,height+top), 20, green, 3)
        cv2.ellipse(new_image, (left + 50,height+top - 10), (60, 10),0,0,360, red, 3)
    
    for person in people_good_distances:
        left, top, width, height = person [: 4]
        #cv2.rectangle (new_image, (left, top), (left + width, top + height), green, 2)
        cv2.ellipse(new_image, (left + 50,height+top - 10), (60, 10), 0, 0,360, green, 3)
    
    if draw_lines:
        for i in range (0, len (people_bad_distances) -1):
            for j in range (i + 1, len (people_bad_distances)):
                cxi, cyi, bevxi, bevyi = people_bad_distances [i] [4:]
                cxj, cyj, bevxj, bevyj = people_bad_distances [j] [4:]
                distance = eucledian_distance ([bevxi, bevyi], [bevxj, bevyj])
                if distance <distance_allowed:
                    cv2.line (new_image, (cxi, cyi), (cxj, cyj), red, 2)
            
    return new_image






#We get the first frame of the video for reference.
original = cv2.cvtColor (cv2.imread ('/home/shreyesh/project/social-distance-detector/frames/frame28.jpg'), cv2.COLOR_BGR2RGB)
image_calibration = original.copy ()
image_copy = original.copy ()

# We manually select 4 points in the image. We use the sidewalk as a reference.
source_points = np.float32 ([[1187, 178], [1575, 220], [933,883], [295, 736]])

print(source_points)

# We draw the selected points in the image:
for point in source_points:
    cv2.circle (image_calibration, tuple (point), 8, (255, 0, 0), -1)

# We draw the connecting lines between the points to form the trapezoid:
points = source_points.reshape ((-1,1,2)). astype (np.int32)
print(points)
cv2.polylines (image_calibration, [points], True, (0,255,0), thickness = 4)

size = (4500, 3000)

src = np.float32 ([[1187, 178], [1575, 220], [933,883], [295, 736]])

dst = np.float32([(0.57, 0.42), (0.65, 0.42), (0.65,0.84), (0.57, 0.84)])

img_size = np.float32([(image_calibration.shape[1], image_calibration.shape[0])])

dst = dst * np.float32(size)


H_matrix = cv2.getPerspectiveTransform(src, dst)

print("The perspective matrix:")
print(H_matrix)


warped = cv2.warpPerspective(image_calibration, H_matrix, size)

plt.figure(figsize=(12,12))
plt.imshow(warped)
plt.xticks(np.arange(0, warped.shape [1], step = 150))
plt.yticks(np.arange(0, warped.shape [0], step=150))
plt.grid(True, color = 'g', linestyle = '-', linewidth = 0.9)
plt.show()


confidence_threshold = 0.5
nms_threshold = 0.4

MIN_DISTANCE = 115

width = 608
height = 608

config = '/home/shreyesh/project/social-distance-detector/yolo-coco/yolov3.cfg'
weights = '/home/shreyesh/project/social-distance-detector/yolo-coco/yolov3.weights'
classes = []

writer = None
W = 2604
H = 1254


with open('/home/shreyesh/project/social-distance-detector/yolo-coco/coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

model = create_model(config, weights)
output_layers = get_output_layers(model)

picture = generate_picture()

video = cv2.VideoCapture('/home/shreyesh/project/social-distance-detector/pedestrians.mp4')

while True:
    
    check, frame = video.read()

    if frame is None:
        break

    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = frame

    blob = blob_from_image(image, (width, height))

    outputs = predict(blob, model, output_layers)

    boxes, nms_boxes, class_ids = non_maximum_suppression(image, outputs, confidence_threshold, nms_threshold)

    person_boxes = get_domain_boxes(classes, class_ids, nms_boxes, boxes, domain_class = 'person')

    good, bad = people_distances_bird_eye_view(person_boxes, MIN_DISTANCE)

    new_image = draw_new_image_with_boxes (image, good, bad, MIN_DISTANCE, draw_lines = True)

    green_points = [g [6:] for g in good]
    red_points = [r [6:] for r in bad]

    bird_eye_view = generate_bird_eye_view (green_points, red_points)
    output_image = generate_content_view (picture, new_image, bird_eye_view)
    output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(output_image, (1280,600))
    cv2.imshow("Image",resized)



    if writer is None:
        fourcc = cv2.VideoWriter_fourcc (* "MJPG")
        writer = cv2.VideoWriter ('/home/shreyesh/project/social-distance-detector/frames/output.avi', fourcc, 30, (W, H), True)
    
    
    if writer is not None:
        writer.write (output_image[:,:, :: - 1])
    

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

if writer is not None:
    writer.release()
video.release()
cv2.destroyAllWindows()




