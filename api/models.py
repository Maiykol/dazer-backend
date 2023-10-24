from django.db import models


class Session(models.Model):
    session_id = models.CharField(max_length=10, primary_key=True)
    
    
class File(models.Model):
    session = models.ForeignKey('Session', on_delete=models.CASCADE, related_name='files')
    filename = models.CharField(max_length=300)
    
    class Meta:
        unique_together = ('session', 'filename')


class Subsampling(models.Model):
    subsample_id = models.CharField(max_length=10, unique=True)
    session = models.ForeignKey('Session', on_delete=models.CASCADE, related_name='subsamples')
    file = models.ForeignKey('File', on_delete=models.CASCADE, related_name='subsamples')
    keep_ratio_columns = models.CharField(max_length=1100)
    target_column = models.CharField(max_length=100)
    
    class Meta:
        unique_together = ('session', 'file', 'subsample_id')

class Classification(models.Model):
    classification_id = models.CharField(max_length=10, unique=True)
    subsample = models.ForeignKey('Subsampling', on_delete=models.CASCADE, related_name='classification')
    