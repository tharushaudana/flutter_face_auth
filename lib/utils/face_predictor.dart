import 'dart:convert';
import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imglib;
import 'image_converter.dart';

class FacePredictor {
  //final int inputDims = 112;
  //final int outputDims = 192;

  final int inputDims = 160;
  final int outputDims = 512;

  Interpreter? _interpreter;
  double threshold = 0.5;

  List _predictedData = [];
  imglib.Image? _currentFaceImage;

  List get predictedData => _predictedData;
  imglib.Image? get currentFaceImage => _currentFaceImage;

  bool initialized = false;

  Future init(VoidCallback cb) async {
    late Delegate delegate;

    try {
      if (Platform.isAndroid) {
        delegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,            
          ),
        );
      } else if (Platform.isIOS) {
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
            allowPrecisionLoss: true,
          ),
        );
      }
      var interpreterOptions = InterpreterOptions()..addDelegate(delegate);

      _interpreter = await Interpreter.fromAsset(
        'assets/facenet_512.tflite',
        options: interpreterOptions,
      );

      initialized = true;
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }

    cb();
  }

  //https://github.com/MCarlomagno/FaceRecognitionAuth/blob/master/lib/services/ml_service.dart

  void reset() {
    _predictedData = [];
    _currentFaceImage = null;
  }

  void captureFace(CameraImage cameraImage, Face? face) {
    if (face == null) throw Exception('Face is null');
    _preProcess(cameraImage, face);
  }

  bool generateCurrentFacePredictionData() {
    if (_currentFaceImage == null) throw Exception('CurrentFaceImage is null');
    if (_interpreter == null) throw Exception('Interpreter is null');

    List input = imageToByteListFloat32(_currentFaceImage!);

    input = input.reshape([1, inputDims, inputDims, 3]);
    List output = List.generate(1, (index) => List.filled(outputDims, 0));

    _interpreter?.run(input, output);
    output = output.reshape([outputDims]);

    _predictedData = List.from(output);

    return true;
  }

  double checkSimilarityWith(List faceData) {
    if (_predictedData.length != faceData.length) throw Exception("Invalid face data");
    return _cosineSimilarity(faceData, _predictedData);
  }

  void printPredictedList() {
    String s = "[";

    for (var v in _predictedData) {
      s += "$v, ";
    }

    s += "]";

    log(s);
  }

  void _preProcess(CameraImage image, Face faceDetected) {
    imglib.Image croppedImage = _cropFace(image, faceDetected);
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, inputDims);
    _currentFaceImage = imglib.copyRotate(img, -90);
  }

  imglib.Image _cropFace(CameraImage image, Face faceDetected) {
    imglib.Image convertedImage = _convertCameraImage(image);
    double x = faceDetected.boundingBox.left - 70.0;
    double y = faceDetected.boundingBox.top - 70.0;
    double w = faceDetected.boundingBox.width + 140.0;
    double h = faceDetected.boundingBox.height + 140.0;
    return imglib.copyCrop(
        convertedImage, x.round(), y.round(), w.round(), h.round());
  }

  imglib.Image _convertCameraImage(CameraImage image) {
    var img = convertToImage(image);
    var img1 = imglib.copyRotate(img, 90);
    return img1;
  }

  Float32List imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * inputDims * inputDims * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < inputDims; i++) {
      for (var j = 0; j < inputDims; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  double _cosineSimilarity(List? e1, List? e2) {
    if (e1 == null || e2 == null) throw Exception("Null argument");

    double mag1 = math.sqrt(e1.map((e) => e * e).reduce((a, b) => a + b));
    double mag2 = math.sqrt(e2.map((e) => e * e).reduce((a, b) => a + b));

    double dot = 0;

    for (int i = 0; i < e1.length; i++) {
      dot += e1[i] * e2[i];
    }

    return dot / (mag1 * mag2);
  }

  double _euclideanDistance(List? e1, List? e2) {
    if (e1 == null || e2 == null) throw Exception("Null argument");

    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += math.pow((e1[i] - e2[i]), 2);
    }
    return math.sqrt(sum);
  }

  close() {
    _interpreter!.close();
  }
}
