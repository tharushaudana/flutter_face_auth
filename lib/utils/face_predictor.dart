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
  final int outputDims = 128;

  Interpreter? _interpreter;
  double threshold = 0.5;

  List _predictedData = [];

  List get predictedData => _predictedData;

  Function(imglib.Image)? onFaceCaptured;

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
        'assets/facenet.tflite',
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

  void setCurrentPrediction(CameraImage cameraImage, Face? face) {
    if (_interpreter == null) throw Exception('Interpreter is null');
    if (face == null) throw Exception('Face is null');
    List input = _preProcess(cameraImage, face);

    input = input.reshape([1, inputDims, inputDims, 3]);
    List output = List.generate(1, (index) => List.filled(outputDims, 0));

    _interpreter?.run(input, output);
    output = output.reshape([outputDims]);

    _predictedData = List.from(output);

    final test = [-0.6477234363555908, -0.3045898675918579, -0.8651089668273926, -0.2966347634792328, -0.004608259070664644, 0.2614383399486542, -0.27486109733581543, 0.16943344473838806, 0.1944338083267212, 0.04334717616438866, -0.21250170469284058, 0.38320106267929077, 0.6687979102134705, -0.39388149976730347, 0.0282137393951416, -0.08371099829673767, 0.1764405369758606, -0.429210901260376, 0.6529709100723267, 0.09092680364847183, -0.4910098910331726, 0.07053028047084808, -0.4019496440887451, 0.7059015035629272, 0.46845996379852295, 0.06061637029051781, 0.6845684051513672, 0.2911202609539032, -0.457896888256073, -0.3459567129611969, -0.2783859670162201, 0.31745463609695435, 0.14051488041877747, 0.5375343561172485, 0.23119845986366272, 0.2525121569633484, -0.09187918901443481, -0.7226192355155945, -0.31496766209602356, -0.3787561357021332, 0.16103409230709076, 1.0954034328460693, -0.5737693309783936, 0.4600409269332886, -0.15805085003376007, -1.099844217300415, 0.38465604186058044, 0.21260324120521545, -0.02888999506831169, 0.4957312047481537, -0.9179791212081909, 0.1854100078344345, -0.3247736394405365, 0.3524324893951416, -0.0036194920539855957, -0.34025856852531433, 0.7265802621841431, -0.19911260902881622, -0.7488769292831421, -0.9275290966033936, 0.25864431262016296, 0.23696686327457428, 0.13716642558574677, 0.5704910755157471, 0.22063308954238892, 0.997248649597168, -0.2087383270263672, 0.7679837346076965, 0.3246099352836609, -0.30355891585350037, 0.18097442388534546, -0.32429516315460205, -0.1011388748884201, 0.2876361012458801, 0.2701660990715027, -0.22905516624450684, -0.18759863078594208, -0.09268583357334137, 0.5597060918807983, 0.12567926943302155, -0.42060011625289917, 0.03230150789022446, -0.27341341972351074, 0.39421728253364563, 1.0249741077423096, 0.5986572504043579, 0.9952332973480225, 0.39063650369644165, -0.673377513885498, 1.0752151012420654, 0.19647115468978882, 0.020053157582879066, -0.11470112204551697, 0.1820170283317566, 0.32820695638656616, 1.011600375175476, -0.3287169635295868, 0.03504199534654617, -1.0575511455535889, 0.3709585964679718, -0.14955133199691772, 0.6403352618217468, 0.027230605483055115, -0.5014301538467407, -0.03835712745785713, -0.24010396003723145, -0.8125787377357483, 0.230394646525383, 0.05655673146247864, -0.8757356405258179, -0.10371118783950806, -0.46632593870162964, 0.32605332136154175, 0.5130228996276855, -0.5416117906570435, 0.6555765867233276, -0.5175007581710815, -0.2317015826702118, 1.1749986410140991, 0.4014393985271454, -0.06813259422779083, 0.22105775773525238, -0.13431291282176971, -0.28645893931388855, 1.208532691001892, -0.9419260621070862, 0.8952643871307373, -0.03284129127860069];

    print("jj: ${test.length}");

    print("Predict Len: ${_predictedData.length}");

    //printPredictedList();

    double dist = _cosineSimilarity(test, predictedData);
    print("Dist: $dist");
  }

  void printPredictedList() {
    String s = "[";

    for (var v in _predictedData) {
      s += "$v, ";
    }

    s += "]";

    log(s);
  }

  List _preProcess(CameraImage image, Face faceDetected) {
    imglib.Image croppedImage = _cropFace(image, faceDetected);
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, inputDims);

    if (onFaceCaptured != null) onFaceCaptured!(img);

    Float32List imageAsList = imageToByteListFloat32(img);
    return imageAsList;
  }

  imglib.Image _cropFace(CameraImage image, Face faceDetected) {
    imglib.Image convertedImage = _convertCameraImage(image);
    double x = faceDetected.boundingBox.left - 10.0;
    double y = faceDetected.boundingBox.top - 10.0;
    double w = faceDetected.boundingBox.width + 10.0;
    double h = faceDetected.boundingBox.height + 10.0;
    return imglib.copyCrop(
        convertedImage, x.round(), y.round(), w.round(), h.round());
  }

  imglib.Image _convertCameraImage(CameraImage image) {
    var img = convertToImage(image);
    var img1 = imglib.copyRotate(img, -90);
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
