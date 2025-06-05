import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from PIL import Image

# Load model at startup
model_path = os.path.join(settings.BASE_DIR, 'fashion_mnist_subset.h5')
model = load_model(model_path)

# Fashion MNIST class names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def index(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(img_file.name, img_file)
        file_path = fs.path(filename)

        # Preprocess image
        img = Image.open(file_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        context['predicted_class'] = predicted_class
        context['filename'] = fs.url(filename)

    return render(request, 'predict.html', context)
