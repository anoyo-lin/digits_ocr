from rest_framework import serializers
from gene.ocr_config import cfg

class TextAnnotationsSerializer(serializers.Serializer):
    description = serializers.CharField()
    prediction = serializers.CharField()
    confidence = serializers.FloatField()
# - prediction CHAR
# - confidence FLOAT
class MNISTTextAnnotationsSerializer(serializers.Serializer):
    prediction = serializers.CharField()
    confidence = serializers.FloatField()
# - order INTEGER
# - top_n DICT
class FullTextAnnotationsSerializer(serializers.Serializer):
    order = serializers.IntegerField()
    top_n = serializers.DictField()

# - textAnnotation MNISTTextAnnotation
# - fullTextAnnotations LIST of FullTextAnnotations
class MNISTOCRResponseSerializer(serializers.Serializer):
    textAnnotations = MNISTTextAnnotationsSerializer()
    fullTextAnnotations = serializers.ListField(
        child=FullTextAnnotationsSerializer()
    )
class RowNumbersOCRResponseSerializer(serializers.Serializer):
    textAnnotations = TextAnnotationsSerializer()
    fullTextAnnotations = serializers.ListField(
        child=FullTextAnnotationsSerializer()
    )
# request body
# - image File
# - confidence_threshold FLOAT
# - top_n INTEGER
class MNISTOCRRequestSerializer(serializers.Serializer):
    image = serializers.FileField(required=True)
    confidence_threshold = serializers.FloatField(
        min_value=cfg.API.MNIST.CONFIDENCE_THRESHOLD.MIN,
        max_value=cfg.API.MNIST.CONFIDENCE_THRESHOLD.MAX,
        default=cfg.API.MNIST.CONFIDENCE_THRESHOLD.DEFAULT,
        required=False
    )

    top_n = serializers.IntegerField(
        min_value=cfg.API.MNIST.TOP_N.MIN,
        max_value=cfg.API.MNIST.TOP_N.MAX,
        default=cfg.API.MNIST.TOP_N.DEFAULT,
        required=False
    )

    def validate_image(self, value):
        if 'image' not in value.content_type:
            raise serializers.ValidationError("Only image files are supported")
        return value
class RowNumbersOCRResponseRequestSerializer(serializers.Serializer):
    image = serializers.FileField(required=True)
    confidence_threshold = serializers.FloatField(
        min_value=cfg.API.MNIST.CONFIDENCE_THRESHOLD.MIN,
        max_value=cfg.API.MNIST.CONFIDENCE_THRESHOLD.MAX,
        default=cfg.API.MNIST.CONFIDENCE_THRESHOLD.DEFAULT,
        required=False
    )

    top_n = serializers.IntegerField(
        min_value=cfg.API.MNIST.TOP_N.MIN,
        max_value=cfg.API.MNIST.TOP_N.MAX,
        default=cfg.API.MNIST.TOP_N.DEFAULT,
        required=False
    )
    def validate_image(self, value):
        if 'image' not in value.content_type:
            raise serializers.ValidationError("Only image files are supported")
        return value