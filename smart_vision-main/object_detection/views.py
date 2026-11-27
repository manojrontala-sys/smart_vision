from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
import os
from .models import ObjectDetect
from image_preprocessor.views import load_image
import tensorflow as tf
import cv2
import numpy as np


# EfficientDet model from TensorFlow Hub
model_url = "https://www.kaggle.com/models/tensorflow/efficientdet/tensorFlow2/d0/1"
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        import tensorflow_hub as hub
        _detector = hub.load(model_url)
    return _detector


def count_objects(image_path, confidence_threshold=0.5):
    detector = get_detector()

    original_image, input_image = load_image(image_path)
    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Perform inference
    detections = detector(input_tensor)

    # Extract bounding boxes and confidence scores
    boxes = detections["detection_boxes"].numpy()[0]  # Bounding boxes
    scores = detections["detection_scores"].numpy()[0]  # Confidence scores

    # Filter out detections with confidence below the threshold
    filtered_indices = np.where(scores > confidence_threshold)[0]
    filtered_boxes = boxes[filtered_indices]

    # Count the number of detected objects
    object_count = len(filtered_boxes)

    # Draw the bounding boxes on the original image
    for box in filtered_boxes:
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (int(xmin * original_image.shape[1]), int(xmax * original_image.shape[1]),
                                      int(ymin * original_image.shape[0]), int(ymax * original_image.shape[0]))
        cv2.rectangle(original_image, (left, top), (right, bottom), (255, 0, 0), 2)  # Draw bounding box with color blue


    # Convert the image back to BGR before saving
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, original_image_bgr)  # Save the image

    return object_count



@login_required(login_url='signin')
def object_detection(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_image.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        object_count = count_objects(image_path)

        object_detect = ObjectDetect.objects.create(
            image='uploads/' + uploaded_image.name,
            object_count=object_count
        )

        context = {
            'object_count': object_count,
            'image_uploaded': True,
            'uploaded_image_name': uploaded_image.name,
            'uploaded_image_url': object_detect.image.url
        }

        return render(request, 'object_detection.html', context)

    return render(request, 'object_detection.html')


@login_required(login_url='signin')
def history(request):
    object_detections = ObjectDetect.objects.all().order_by('-uploaded_at')
    return render(request, 'object_detection_history.html', {'object_detections': object_detections})


@login_required(login_url='signin')
def delete(request, pk):
    object_detect = get_object_or_404(ObjectDetect, id=pk)
    try:
        if object_detect.image and object_detect.image.name:
            if os.path.exists(object_detect.image.path):
                os.remove(object_detect.image.path)
    except Exception as e:
        pass

    object_detect.delete()

    return redirect('object_detection_history')