#from django.shortcuts import render

from django.shortcuts import redirect
from drf_yasg.openapi import Parameter
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import FormParser, MultiPartParser


import tensorflow as tf 
import numpy as np
from cv2 import cv2



# Create your views here.

from gene import ocr_config
from gene.ocr_config import cfg
from gene.digits_ocr.ocr_response import MNISTOCRResponse
from gene.digits_ocr.serializers import MNISTOCRResponseSerializer, MNISTOCRRequestSerializer
from gene.digits_ocr.utils import predict


@api_view(['GET'])
def clear_session(request):
    if request.method == 'GET':
        tf.keras.backend.clear_session()
        return Response(['Session Cleared'])

def redirect_home_view(request):
    response = redirect('/swagger/')
    return response

class TFVersionView(APIView):
    """
    get: TF related info
    """

    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        """
        get: Tensorflow related setup info on the running instance
        """
        return Response([{
            'tensorflow version': tf.__version__,
            'is_built_with_cuda': tf.test.is_built_with_cuda(),
            'is_gpu_available': tf.test.is_gpu_available()
        }])

class MNISTView(APIView):
    """
    POST: perform OCR on single digit image
    """
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        request_body=MNISTOCRRequestSerializer,
        operation_summary='Perfrom OCR on a single digit',
        operation_description='Perfrom OCR on request image and return prediction and confidence score of prediciton',
        manual_parameters=[
            Parameter(
                name='image', required=True, in_='formData', type='file',
                description='image(supported format: TIFF, JPEG, PNG) for OCR'
                ),
            Parameter(
                name='confidence_threshold', in_='formData', type='float',
                description='Predicitons below confidence threshold would be masked by "*"\n Range 0.0-1.0',
                default=cfg.API.MNIST.CONFIDENCE_THRESHOLD.DEFAULT
                ),
            Parameter(
                name='top_n', in_='formData', type='integer',
                description='Top N predictions to be returned on each characters in image\n Range 0-12',
                default=cfg.API.MNIST.TOP_N.DEFAULT
            )

        ],
        responses={200: MNISTOCRResponseSerializer, 400: '400 Bad Request'}
    )

    def post(self, request, *args, **kwargs):
        request_serializer = MNISTOCRRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        img_data = request.data['image']
        # parse the image data into numpy array
        img = np.asarray(bytearray(img_data.read()), np.uint8)
        # convert 256-level grayScale
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        
        # normalize
        if np.mean(img) > 128:
            # in mnist 0 is white 255 is black
            # if main backgroud color is black, flip it
            temp = 255 - cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            temp = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        digits_images = np.asarray(temp) / 255.0
        mnist_model = tf.keras.models.load_model(ocr_config.cfg.MNIST.MODEL_TO_USE)
        results = predict(digits_images, mnist_model)
        response = MNISTOCRResponse(
            results,
            confidence_threshold=request_serializer.data['confidence_threshold'],
            top_n=request_serializer.data['top_n']
        )
        response_serializer = MNISTOCRResponseSerializer(response)
        return Response([response_serializer.data])