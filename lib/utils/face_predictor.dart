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
  Interpreter? _interpreter;
  double threshold = 0.5;

  List _predictedData = [];
  List get predictedData => _predictedData;

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
        'assets/mobilefacenet.tflite',
        options: interpreterOptions,
      );

      initialized = true;
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }

    cb();
  }

  final test = [-0.00884301122277975, 0.042804401367902756, 0.016355594620108604, -0.007011133711785078, -0.06902765482664108, 0.057357072830200195, -0.055933982133865356, -0.026272471994161606, -0.08825168013572693, -0.10911130905151367, -0.01916610635817051, 0.009055777452886105, -0.008292263373732567, 0.0410006120800972, -0.002883587032556534, -0.09165322035551071, -0.029617397114634514, -0.013291389681398869, 0.00024197310267481953, 0.0010570683516561985, -0.16131775081157684, 0.0770675465464592, -0.0748332142829895, 0.017003130167722702, 0.027055148035287857, 0.016276150941848755, -0.05532093346118927, 0.0783647894859314, 0.19760875403881073, -0.05390816554427147, -0.007313713897019625, 0.19006454944610596, 0.1384902149438858, 0.0006404858431778848, -0.1129794791340828, 0.05342930182814598, -0.010493838228285313, -0.011447637341916561, 0.0009177750325761735, 0.029282350093126297, 0.010480434633791447, 0.005238309036940336, 0.017498357221484184, -0.009155015461146832, 0.012412902899086475, -0.04715631902217865, -0.09902207553386688, -0.008705156855285168, -0.010318410582840443, 0.0716596469283104, 0.06760989874601364, -0.008283054456114769, -0.2890791893005371, -0.004822397604584694, -0.03130919113755226, 0.012250750325620174, 0.04816028103232384, 0.004904256202280521, -0.11200977116823196, 0.03582892566919327, 0.05182855576276779, -0.11623966693878174, -0.02452937886118889, 0.10320964455604553, -0.015374361537396908, 0.010289257392287254, -0.008507047779858112, 0.008107729256153107, 0.008174438029527664, 0.00391434459015727, -0.042177051305770874, -0.10649818181991577, 0.0005402752431109548, 0.015961306169629097, -0.06784527003765106, 0.013929393142461777, 0.0017534487415105104, -0.00009202975343214348, 0.24856674671173096, -0.015143655240535736, -0.00860572885721922, 0.013433689251542091, -0.01342072430998087, 0.15412291884422302, -0.029436834156513214, -0.002194970613345504, -0.0011816952610388398, 0.0125389089807868, 0.0027541201561689377, -0.055878762155771255, 0.03709400072693825, -0.0005365388351492584, 0.012394611723721027, -0.02915082685649395, -0.07655376195907593, -0.12530595064163208, -0.03221362084150314, -0.0162954218685627, -0.00033990517840720713, 0.009959319606423378, 0.003570032771676779, -0.013246297836303711, -0.007697570137679577, -0.002910420298576355, -0.0023341167252510786, 0.007257359102368355, -0.19322852790355682, 0.0005536543321795762, 0.003999659325927496, 0.02654801681637764, -0.19501399993896484, 0.015139699913561344, 0.011162669397890568, 0.23838482797145844, 0.015813222154974937, 0.07667510956525803, -0.005669306963682175, -0.03825521841645241, 0.08798746764659882, 0.16905993223190308, 0.20751065015792847, -0.01978316716849804, -0.120706707239151, -0.002836747793480754, 0.0009635742171667516, -0.0031464118510484695, 0.005177782848477364, 0.0006496681016869843, 0.006503606680780649, -0.07389640063047409, 0.014137760736048222, 0.030192460864782333, -0.0015996824949979782, 0.018732748925685883, 0.03524501994252205, -0.0153612419962883, -0.19126535952091217, -0.06346649676561356, 0.0037326174788177013, 0.008115229196846485, -0.006114281713962555, 0.00009818696707952768, 0.0007231557974591851, -0.05062282085418701, -0.18180420994758606, 0.10201345384120941, -0.021521534770727158, -0.00002311524258402642, 0.014601435512304306, -0.011042767204344273, -0.015607195906341076, -0.09084396809339523, -0.04025673121213913, -0.03558974340558052, -0.007369674276560545, -0.00028385515906848013, 0.007059725001454353, 0.017242494970560074, -0.12252666801214218, -0.002620188519358635, -0.07364407926797867, -0.0005053795175626874, -0.013009299524128437, 0.00040921856998465955, 0.0068176123313605785, 0.023760555312037468, 0.02210952714085579, -0.0898304134607315, -0.004325610585510731, 0.00006516966095659882, 0.17882508039474487, 0.04204007610678673, -0.0031216975767165422, -0.06126813217997551, 0.01575644500553608, 0.005790725816041231, -0.08116395771503448, -0.08916166424751282, -0.013093733228743076, 0.006694413721561432, -0.03856474533677101, -0.004700345918536186, 0.0016910042613744736, -0.006297909654676914, 0.10934305191040039, 0.02825477160513401, -0.08984614163637161, 0.0921567901968956, 0.18112100660800934, -0.10492914915084839, -0.07964607328176498, -0.015710560604929924];

  void setCurrentPrediction(CameraImage cameraImage, Face? face) {
    if (_interpreter == null) throw Exception('Interpreter is null');
    if (face == null) throw Exception('Face is null');
    List input = _preProcess(cameraImage, face);

    input = input.reshape([1, 112, 112, 3]);
    List output = List.generate(1, (index) => List.filled(192, 0));

    _interpreter?.run(input, output);
    output = output.reshape([192]);

    _predictedData = List.from(output);

    //print(base64Encode(_predictedData.whereType<int>().toList()));

    print("jj: ${test.length}");

    print("Predict Len: ${_predictedData.length}");

    double dist = _euclideanDistance(test, predictedData);

    print("Dist: $dist");

    //printPredictedList();
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
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, 112);

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
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  double _euclideanDistance(List? e1, List? e2) {
    if (e1 == null || e2 == null) throw Exception("Null argument");

    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += math.pow((e1[i] - e2[i]), 2);
    }
    return math.sqrt(sum);
  }
}
