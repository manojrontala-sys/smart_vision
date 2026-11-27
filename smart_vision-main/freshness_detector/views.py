from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from .models import FreshnessPrediction
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import time

# Define the classes
all_classes = ['FreshApple', 'RottenApple', 'FreshBanana', 'RottenBanana', 'FreshMango', 'RottenMango',
               'FreshOrange', 'RottenOrange', 'FreshStrawberry', 'RottenStrawberry', 'FreshCarrot', 'RottenCarrot',
               'FreshTomato', 'RottenTomato', 'FreshCucumber', 'RottenCucumber', 'FreshPotato', 'RottenPotato',
               'FreshBellpepper', 'RottenBellpepper']

model_path = os.path.join(settings.BASE_DIR, 'freshness_detector', 'model', 'fruit_veg_freshness_model.h5')
_model = None


def _load_freshness_model_from_h5_weights(filepath):
    """Rebuild the MobileNetV2 classifier and load weights from HDF5 by layer name."""
    import h5py

    base_model = tf.keras.applications.MobileNetV2(
        weights=None, include_top=False, input_shape=(150, 150, 3)
    )
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', name='dense'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(all_classes), activation='softmax', name='dense_1'),
    ])
    model.build((None, 150, 150, 3))

    with h5py.File(filepath, 'r') as f:
        mw = f['model_weights']
        backbone_weights = mw['mobilenetv2_1.00_224']
        for layer in base_model.layers:
            if layer.weights and layer.name in backbone_weights:
                grp = backbone_weights[layer.name]
                layer.set_weights([grp[w.name][:] for w in layer.weights])
        for layer_name in ('dense', 'dense_1'):
            layer = model.get_layer(layer_name)
            grp = mw[layer_name]['sequential'][layer_name]
            layer.set_weights([grp[w.name][:] for w in layer.weights])

    return model


def get_model():
    global _model
    if _model is None:
        for loader in (
            lambda: __import__('tf_keras').models.load_model(model_path, compile=False),
            lambda: tf.keras.models.load_model(model_path, compile=False),
            lambda: _load_freshness_model_from_h5_weights(model_path),
        ):
            try:
                _model = loader()
                break
            except Exception:
                continue
        if _model is None:
            _model = _load_freshness_model_from_h5_weights(model_path)
    return _model


def predict_image(image_path):
    start_time = time.time()
    model = get_model()

    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    predicted_class = all_classes[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]

    end_time = time.time()
    time_taken = end_time - start_time

    return predicted_class, confidence_score, time_taken



@login_required(login_url='signin')
def freshness_detector(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_image.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        predicted_class, confidence_score, time_taken = predict_image(image_path)

        prediction = FreshnessPrediction.objects.create(
            image='uploads/' + uploaded_image.name,
            predicted_class=predicted_class,
            confidence_score=confidence_score,
            time_taken=time_taken
        )

        context = {
            'predicted_class': predicted_class,
            'confidence_score': float(confidence_score),
            'time_taken': round(float(time_taken), 4),
            'image_uploaded': True,
            'uploaded_image_url': prediction.image.url,
            'uploaded_image_name': uploaded_image.name
        }

        return render(request, 'freshness_detector.html', context)

    return render(request, 'freshness_detector.html')


@login_required(login_url='signin')
def history(request):
    predictions = FreshnessPrediction.objects.all().order_by('-uploaded_at')
    return render(request, 'freshness_detector_history.html', {'predictions': predictions})


@login_required(login_url='signin')
def delete(request, pk):
    prediction = get_object_or_404(FreshnessPrediction, id=pk)
    try:
        if prediction.image and prediction.image.name:
            if os.path.exists(prediction.image.path):
                os.remove(prediction.image.path)
    except Exception as e:
        pass

    prediction.delete()

    return redirect('freshness_detector_history')
