sleep 10
celery -A dazer_backend worker -l info -P threads -c 8