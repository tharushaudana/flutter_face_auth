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

  //https://github.com/MCarlomagno/FaceRecognitionAuth/blob/master/lib/services/ml_service.dart

  void setCurrentPrediction(CameraImage cameraImage, Face? face) {
    if (_interpreter == null) throw Exception('Interpreter is null');
    if (face == null) throw Exception('Face is null');
    List input = _preProcess(cameraImage, face);

    input = input.reshape([1, 112, 112, 3]);
    List output = List.generate(1, (index) => List.filled(192, 0));

    _interpreter?.run(input, output);
    output = output.reshape([192]);

    _predictedData = List.from(output);

    final test = [0.00034738227259367704, 0.03294375538825989, 0.016338014975190163, 0.0033659585751593113, -0.07208213210105896, 0.05389386788010597, -0.012093165889382362, -0.03789576515555382, 0.02234053798019886, -0.186056450009346, -0.028449393808841705, 0.00844119768589735, -0.00711802626028657, 0.029565272852778435, -0.001243395498022437, -0.08203097432851791, -0.04058096930384636, -0.019771603867411613, -0.0048567201010882854, 0.012992937117815018, -0.2137996107339859, 0.05683575198054314, -0.048470769077539444, 0.020236104726791382, 0.09829595685005188, 0.008594995364546776, -0.06137873977422714, 0.08565351366996765, 0.13835486769676208, -0.02760721743106842, -0.004035561345517635, 0.16493827104568481, 0.09230902045965195, 0.0014654049882665277, 0.06267806887626648, 0.05836312100291252, -0.12227339297533035, -0.0042551439255476, 0.005270053632557392, -0.038375064730644226, 0.015382695943117142, 0.007180432789027691, 0.02178938500583172, -0.014699890278279781, 0.015480185858905315, -0.030666440725326538, -0.08245289325714111, 0.006410697940737009, -0.01707730069756508, 0.0675768032670021, 0.0012160937767475843, -0.009475701488554478, -0.16896547377109528, -0.005971399135887623, -0.10621442645788193, 0.00701178191229701, 0.08646449446678162, 0.004288526251912117, -0.1239176094532013, 0.03534502536058426, 0.06493964046239853, -0.14054091274738312, -0.1630711555480957, -0.015088292770087719, -0.02819526568055153, 0.04308468475937843, -0.008883477188646793, 0.015344924293458462, 0.007950898259878159, 0.005442709196358919, -0.03786696121096611, -0.08324124664068222, -0.08794264495372772, 0.0164723489433527, -0.04581085219979286, 0.0019190877210348845, 0.005287936422973871, -0.0017155418172478676, 0.10838102549314499, -0.07705137133598328, -0.012642516754567623, 0.04540112614631653, -0.00814106035977602, 0.2579420804977417, 0.010381316766142845, -0.0029191409703344107, -0.010197688825428486, -0.058694321662187576, -0.029307490214705467, -0.14162106812000275, 0.07268131524324417, -0.0038251406513154507, 0.009408391080796719, -0.03487627953290939, -0.07916165888309479, -0.2161136418581009, -0.022176947444677353, -0.15780039131641388, -0.0013970265863463283, 0.010734674520790577, 0.005278979893773794, -0.020312698557972908, -0.01516687497496605, 0.00005273105125525035, -0.0014170558424666524, 0.008433341979980469, -0.18281632661819458, 0.007227059453725815, -0.0000376379830413498, 0.023844847455620766, -0.06422875076532364, 0.012055824510753155, 0.011609240435063839, 0.2166915386915207, 0.008988931775093079, 0.0739741250872612, -0.0019506511744111776, -0.050719279795885086, -0.022893331944942474, 0.1093791276216507, 0.15596984326839447, -0.013429122045636177, -0.023830506950616837, -0.005238482262939215, 0.005846103653311729, -0.0018059773137792945, 0.008866299875080585, -0.0022499007172882557, -0.0038063297979533672, -0.05627748742699623, 0.011792483739554882, 0.03692155331373215, -0.0005413699545897543, -0.02485325001180172, -0.018077407032251358, -0.012159843929111958, -0.15621626377105713, 0.022174851968884468, 0.006416940595954657, 0.007435414008796215, -0.008226611651480198, 0.0010504701640456915, -0.0030274747405201197, -0.051805805414915085, -0.14725393056869507, 0.17763154208660126, -0.030433474108576775, 0.0009374918881803751, 0.015097612515091896, -0.006238831207156181, -0.012944032438099384, -0.1335039883852005, -0.0379292294383049, -0.03511127084493637, -0.008897286839783192, 0.00234216614626348, 0.004640903789550066, 0.00920675229281187, -0.203287273645401, -0.0002479625982232392, -0.08296409994363785, -0.002141772536560893, -0.009799429215490818, 0.00046091628610156476, 0.007966979406774044, 0.02242172509431839, 0.017522914335131645, 0.08878772705793381, -0.002601467538625002, -0.0063112229108810425, 0.2365485429763794, 0.059294410049915314, -0.0023867764975875616, 0.08131484687328339, 0.04156394302845001, 0.006592176854610443, -0.02877039648592472, -0.06705076992511749, -0.014233503490686417, 0.00953610148280859, 0.011001741513609886, 0.015512700192630291, -0.0059442175552248955, -0.004909148905426264, 0.15756012499332428, 0.0226041991263628, -0.09638242423534393, 0.025915779173374176, 0.16987824440002441, -0.04414821043610573, -0.0798054113984108, -0.017684584483504295];

    print("jj: ${test.length}");

    print("Predict Len: ${_predictedData.length}");

    //printPredictedList();

    double dist = _euclideanDistance(test, predictedData);

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

  close() {
    _interpreter!.close();
  }
}
