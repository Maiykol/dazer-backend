from django.contrib import admin

# Register your models here.
from .models import Session, File, Subsampling, ClassificationTask

admin.site.register(Session)
admin.site.register(File)
admin.site.register(Subsampling)
admin.site.register(ClassificationTask)
