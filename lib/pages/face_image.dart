import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:image/image.dart' as imglib;

class FaceImage extends StatefulWidget {
  final imglib.Image image;

  const FaceImage({
    super.key,
    required this.image,
  });

  @override
  State<StatefulWidget> createState() => _FaceImageState();
}

class _FaceImageState extends State<FaceImage> {
  imglib.Image? imglibImage;
  ui.Image? uiImage;

  void loadImageToUiImage(imglib.Image image) async {
    imglibImage = image;
    uiImage = null;

    ui.decodeImageFromPixels(
      image.getBytes(),
      image.width,
      image.height,
      ui.PixelFormat.rgba8888,
      (ui.Image img) {
        setState(() {
          uiImage = img;
        });
      },
    );

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    if (imglibImage != widget.image) loadImageToUiImage(widget.image);

    if (uiImage == null) {
      return const CircularProgressIndicator();
    }

    return CustomPaint(
      painter: _UIImagePainter(uiImage!),
      size: Size(uiImage!.width.toDouble(), uiImage!.height.toDouble()),
    );
  }
}

class _UIImagePainter extends CustomPainter {
  final ui.Image image;

  _UIImagePainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint(),
    );
  }

  @override
  bool shouldRepaint(_UIImagePainter oldDelegate) {
    return image != oldDelegate.image;
  }
}