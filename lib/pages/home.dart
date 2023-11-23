import 'dart:async';
import 'dart:developer';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:face_auth/utils/face_predictor.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'dart:ui' as ui;
import 'package:image/image.dart' as imglib;
import 'package:face_auth/utils/image_converter.dart';

class HomePage extends StatefulWidget {
  const HomePage({
    super.key,
    required this.cameras,
  });

  final List<CameraDescription> cameras;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late CameraController cameraController;
  late FaceDetector faceDetector;
  late CameraDescription camera;

  final facePredictor = FacePredictor();

  CameraImage? currentImage;

  ui.Image? currentFaceImage;

  bool disableImageUpdate = false;

  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  initFaceDetector() {
    faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableLandmarks: true,
        performanceMode: FaceDetectorMode.accurate,
      ),
    );
  }

  InputImage? convertToInputImage(CameraImage image) {
    final sensorOrientation = camera.sensorOrientation;
    InputImageRotation? rotation;

    var rotationCompensation =
        _orientations[cameraController.value.deviceOrientation];

    if (rotationCompensation == null) {
      log("rotationCompensation is NULL !!!!!!!!!!!");
      return null;
    }

    if (camera.lensDirection == CameraLensDirection.front) {
      // front-facing
      rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
    } else {
      // back-facing
      rotationCompensation =
          (sensorOrientation - rotationCompensation + 360) % 360;
    }

    rotationCompensation = 0;

    rotation = InputImageRotationValue.fromRawValue(rotationCompensation);

    if (rotation == null) {
      log("Rotation is NULL !!!!!!!!!!!");
      return null;
    }

    // get image format
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    // validate format depending on platform
    // only supported formats:
    // * nv21 for Android
    // * bgra8888 for iOS

    /*if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      log("Invalid image format !!!!!!!!!!!");
      return null;
    }

    // since format is constraint to nv21 or bgra8888, both only have one plane
    if (image.planes.length != 1) {
      log("So many planes !!!!!!!!!!!");
      return null;
    }*/

    final plane = image.planes.first;

    // compose InputImage using bytes
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation, // used only in Android
        format: format!, // used only in iOS
        bytesPerRow: plane.bytesPerRow, // used only in iOS
      ),
    );
  }

  processImage() async {
    if (currentImage == null) {
      log("Current Image is NULL !!!!!!!!!!!!");
      return;
    }

    InputImage? inputImage = convertToInputImage(currentImage!);

    if (inputImage == null) return;

    final List<Face> faces = await faceDetector.processImage(inputImage);

    log("Faces detected: ${faces.length}");

    if (faces.length != 1) return;

    facePredictor.setCurrentPrediction(currentImage!, faces[0]);

    /*for (Face face in faces) {
      final Rect boundingBox = face.boundingBox;

      final double? rotX =
          face.headEulerAngleX; // Head is tilted up and down rotX degrees
      final double? rotY =
          face.headEulerAngleY; // Head is rotated to the right rotY degrees
      final double? rotZ =
          face.headEulerAngleZ; // Head is tilted sideways rotZ degrees

      // If landmark detection was enabled with FaceDetectorOptions (mouth, ears,
      // eyes, cheeks, and nose available):
      final FaceLandmark? leftEar = face.landmarks[FaceLandmarkType.leftEar];
      if (leftEar != null) {
        final Point<int> leftEarPos = leftEar.position;
      }

      // If classification was enabled with FaceDetectorOptions:
      if (face.smilingProbability != null) {
        final double? smileProb = face.smilingProbability;
      }

      // If face tracking was enabled with FaceDetectorOptions:
      if (face.trackingId != null) {
        final int? id = face.trackingId;
      }
    }*/
  }

  startStream() {
    cameraController.startImageStream((CameraImage image) {
      if (disableImageUpdate) return;
      currentImage = image;
    });
  }

  Future<ui.Image> loadImageToUiImage(imglib.Image image) async {
    Completer<ui.Image> completer = Completer();
    ui.decodeImageFromPixels(
      image.getBytes(),
      image.width,
      image.height,
      ui.PixelFormat.rgba8888,
      (ui.Image img) {
        completer.complete(img);
      },
    );
    return completer.future;
  }

  @override
  void initState() {
    camera = widget.cameras[1];

    super.initState();

    cameraController = CameraController(
      camera,
      ResolutionPreset.max,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.nv21 // for Android
          //? ImageFormatGroup.bgra8888 // for Android
          : ImageFormatGroup.bgra8888,
    );

    cameraController.initialize().then((_) {
      if (!mounted) {
        log("Not mounted...!!!!!!!!!!!!!");
        return;
      }

      log("Camera ready..........ok");

      startStream();

      setState(() {});
    }).catchError((Object e) {
      if (e is CameraException) {
        switch (e.code) {
          case 'CameraAccessDenied':
            log("Access Denided...!!!!!!!!!!!!!");
            break;
          default:
            // Handle other errors here.
            break;
        }
      }
    });

    facePredictor.init(() {
      setState(() {});
    });

    facePredictor.onFaceCaptured = (img) async {
      currentFaceImage = await loadImageToUiImage(img);
      setState(() {
        
      });
    };

    initFaceDetector();
  }

  @override
  void dispose() {
    try {
      cameraController.stopImageStream();
      cameraController.dispose();
      faceDetector.close();
      facePredictor.close();
    } catch (e) {}

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text("Face Auth"),
      ),
      body: Center(
        child: !cameraController.value.isInitialized || !facePredictor.initialized
            ? Text("Not initialized yet")
            : currentFaceImage == null ?  Stack(
                children: [
                  CameraPreview(cameraController),
                  Align(
                    alignment: Alignment.bottomCenter,
                    child: FilledButton(
                      onPressed: () {
                        processImage();
                      },
                      child: const Text("PROCESS"),
                    ),
                  )
                ],
              ) : Column(
                children: [
                  UIImage(image: currentFaceImage!),
                  FilledButton(onPressed: () {
                    setState(() {
                      currentFaceImage = null;
                    });
                  }, child: Text("Reset"),),
                ],
              )
      ),
    );
  }
}

class UIImage extends StatelessWidget {
  final ui.Image image;

  const UIImage({
    super.key,
    required this.image,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _UIImagePainter(image), size: Size(image.width.toDouble(), image.height.toDouble()),);
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