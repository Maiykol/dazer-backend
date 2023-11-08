from django.db import models


class Session(models.Model):
    session_id = models.CharField(max_length=10, primary_key=True)
    
    
class File(models.Model):
    session = models.ForeignKey('Session', on_delete=models.CASCADE, related_name='files')
    filename = models.CharField(max_length=100)
    rows_removed = models.IntegerField(default=0) # some rows are dropped when containing NaN. 0 if no row removed
    columns = models.TextField()
    categorical_columns_values = models.TextField()
    
    class Meta:
        unique_together = ('session', 'filename')


class Subsampling(models.Model):
    subsample_id = models.CharField(max_length=10, unique=True)
    session = models.ForeignKey('Session', on_delete=models.CASCADE, related_name='subsamples')
    file = models.ForeignKey('File', on_delete=models.CASCADE, related_name='subsamples')
    keep_ratio_columns = models.TextField()
    test_ratio = models.FloatField()
    allowed_deviation = models.FloatField()
    ratios = models.TextField()
    iteration_random_states = models.TextField()
    result_formatted = models.TextField()
    
    class Meta:
        unique_together = ('session', 'file', 'subsample_id')


class ClassificationTask(models.Model):
    classification_task_id = models.CharField(max_length=10, unique=True)
    subsample = models.ForeignKey('Subsampling', on_delete=models.CASCADE, related_name='classification')
    status = models.CharField(max_length=300)
    evaluation = models.TextField() # stringified list of dictionaries
    feature_importances = models.TextField()
    feature_columns = models.TextField()
    target_column = models.CharField(max_length=100)
    target_value = models.CharField(max_length=100)
    cv = models.IntegerField()
    random_states = models.CharField(max_length=100)
