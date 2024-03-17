from django.contrib import admin
from django.urls import include, path
from image_classify_app import views


urlpatterns = [
    path("", views.img_classify, name = "img_classify" )
]