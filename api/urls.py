from django.urls import path
from api import views

urlpatterns = [
    path('file_upload/<str:session>/<str:filename>', views.FileUpload.as_view()),
    path('file/<str:subsample_id>', views.File.as_view()),
    path('file_columns_subsampling/<str:subsample_id>', views.FileColumnsSubsample.as_view()),
    path('file_columns/<str:session>/<str:filename>', views.FileColumns.as_view()),
    path('session_id/', views.SessionId.as_view()),
    path('session_files/<str:session>', views.SessionFiles.as_view()),
    path('subsample/result/<str:subsample_id>', views.SubsampleResult.as_view()),
    path('subsample/<str:session>/<str:filename>', views.Subsample.as_view()),
    path('classification/<str:subsample_id>', views.Classification.as_view()),
    path('classification_result/<str:classification_task_id>', views.ClassificationResult.as_view()),
    
    path('delete/classification/<str:classification_task_id>', views.ClassificationDelete.as_view()),
    path('delete/subsample/<str:subsample_id>', views.SubsampleDelete.as_view()),
    path('delete/file/<str:session>/<str:filename>', views.FileDelete.as_view()),
    path('delete/session/<str:session>', views.SessionDelete.as_view()),
    
    
]

