# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, Response
from flask_login import login_required
from jinja2 import TemplateNotFound

from object_detection.detector import YoloObjectDetection


@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')


# @blueprint.route('/visual-pattern')
# @login_required
# def visual_pattern():
#     print("ertfgdfg dfgdg df gg dfgdf")
#     return render_template('home/visual-pattern.html', segment='index'), 10


@blueprint.route('/video_feed/')
def video_feed():
    video_url = request.args['url']
    type_ = request.args['type_']
    cam_name = request.args['cam_name']
    video = YoloObjectDetection(video_url, type_, cam_name)
    return Response(video.get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @blueprint.route('/stats/')
# def stats():
#     print(f"-- {request.args['url']=}")
#     video = YoloObjectDetection(request.args['url'])
#     return str(video.objects_count)


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)
        print(f"{segment=}")
        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except Exception as e:
        print(f"{e}")
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
