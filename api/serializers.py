from rest_framework import serializers
from api import models
import json


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.File
        fields = ['filename', 'rows_removed']
        
class TaskStatusSerializer(serializers.ModelSerializer):
    status = serializers.SerializerMethodField()
    
    def get_status(self, obj):
        return json.loads(obj.status)
     
    class Meta:
        model = models.Task
        fields = ['status', 'progress']
        
class ClassificationTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ClassificationTask
        field = ['__all__']