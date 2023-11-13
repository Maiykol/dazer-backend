import os

from celery import Celery


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dazer_backend.settings")

app = Celery("dazer_backend")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()