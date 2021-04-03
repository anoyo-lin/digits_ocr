"""gene URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions
from django.conf.urls import url
from django import urls

from django.conf.urls import include

from gene.digits_ocr import views

schema_view = get_schema_view(
    openapi.Info(
        title="Handwritten Number OCR API",
        default_version='v1',
        description="A OCR API can predict handwrittent number by NMIST",
        contact=openapi.Contact("anoyo_lin@qq.com"),
        license=openapi.License("BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
    
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('django_prometheus.urls')),
    path('', views.redirect_home_view, name='gene_test'),
    path('api/ocr/mnist', views.MNISTView.as_view(), name='mnist_ocr'),
    path('api/ocr/tensorflowinfo', views.TFVersionView.as_view(), name='tensorflow_info'),
    url(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    url(r'^swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    url(r'^redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
