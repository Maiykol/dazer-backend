from django.urls import path
from api import views

urlpatterns = [
    path('file_upload/<str:session>/<str:filename>', views.FileUpload.as_view()),
    path('file_columns/<str:session>/<str:filename>', views.FileColumns.as_view()),
    path('session_id/', views.SessionId.as_view()),
    path('session_files/<str:session>', views.SessionFiles.as_view()),
    path('subsample/<str:session>/<str:filename>', views.Subsample.as_view()),
    path('subsample_result/<str:subsample_id>', views.SubsampleResult.as_view())
]

