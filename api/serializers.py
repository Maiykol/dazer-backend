from rest_framework import serializers
from api import models


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.File
        fields = ['filename', 'rows_removed']
        
class ClassificationTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ClassificationTask
        field = ['__all__']