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

    final test = [-0.00024537520948797464, 0.0375073067843914, 0.016411520540714264, 0.005773976910859346, -0.07302340865135193, 0.04727598652243614, 0.017860623076558113, -0.06089917570352554, 0.01570812240242958, -0.11859701573848724, -0.021403715014457703, 0.010176287963986397, -0.008358489722013474, 0.03772994130849838, -0.0019900689367204905, -0.11695936322212219, -0.04575245454907417, -0.01888359524309635, -0.004380702041089535, 0.009002886712551117, -0.2083018571138382, 0.07557269930839539, -0.05343455821275711, 0.018862923607230186, 0.09935988485813141, 0.016791703179478645, -0.06116180866956711, 0.12227489054203033, 0.2005912810564041, -0.0504656657576561, -0.013121376745402813, 0.18118031322956085, 0.12000403553247452, 0.003734838915988803, 0.05633387342095375, 0.03177197277545929, -0.08866667002439499, -0.005504239816218615, 0.004223180934786797, 0.022840697318315506, 0.01437263935804367, 0.00681706378236413, 0.020340951159596443, -0.011528987437486649, 0.017215821892023087, -0.036984436213970184, -0.08277972787618637, -0.02407107502222061, -0.017021844163537025, 0.06135551631450653, 0.012422445230185986, -0.009317856281995773, -0.21943464875221252, -0.0062728640623390675, -0.04547153785824776, 0.010999036021530628, 0.07075788080692291, 0.005078786984086037, -0.13190269470214844, 0.03527923673391342, 0.06543353199958801, -0.14790086448192596, -0.17625334858894348, -0.016540981829166412, -0.028206128627061844, -0.03178166598081589, -0.00997516792267561, -0.006436723284423351, 0.007906789891421795, 0.003989657387137413, -0.04123286157846451, -0.010648448020219803, -0.004864174872636795, 0.017203008756041527, -0.04256370663642883, -0.00020860350923612714, 0.003175614168867469, -0.0012235670583322644, 0.15670932829380035, -0.07178603857755661, -0.012818118557333946, 0.0692390650510788, -0.013927335850894451, 0.22254882752895355, -0.009006237611174583, -0.0029365401715040207, -0.010647102259099483, -0.023854460567235947, -0.03503987938165665, -0.15412351489067078, 0.09751790016889572, -0.0036583696492016315, 0.012422292493283749, -0.02975054830312729, -0.001664192765019834, -0.1392734795808792, -0.017863769084215164, -0.03234704211354256, 0.0020445752888917923, 0.00869050994515419, 0.004008342977613211, -0.020513858646154404, -0.013810912147164345, -0.0005523294676095247, -0.000903016421943903, 0.007800758816301823, -0.1251598298549652, 0.005570251028984785, 0.00528867170214653, 0.025182414799928665, -0.0826406329870224, 0.010942547582089901, 0.012956175021827221, 0.26339083909988403, 0.01040805783122778, 0.08846337348222733, -0.0010865324875339866, -0.04497337341308594, 0.08238841593265533, 0.13008317351341248, 0.1886739879846573, -0.016064278781414032, -0.1373947113752365, -0.0034384008031338453, 0.003073721658438444, -0.0002672213886398822, 0.00521480105817318, 0.004700814839452505, -0.0024838545359671116, -0.08751785010099411, 0.011666045524179935, 0.031966451555490494, 0.00017924226995091885, -0.037692390382289886, 0.015608013607561588, -0.014695491641759872, -0.19186465442180634, 0.010771221481263638, 0.013084804639220238, 0.006232546642422676, -0.008517296984791756, 0.0009315802017226815, -0.0031739259138703346, -0.06291502714157104, -0.1723630130290985, 0.14056792855262756, -0.034035634249448776, 0.0014123357832431793, 0.01266262587159872, -0.00744639104232192, -0.011313478462398052, -0.10709436982870102, -0.009737357497215271, -0.038356050848960876, -0.0086427116766572, 0.0019060098566114902, 0.0018221704522147775, 0.006076055113226175, -0.15520934760570526, -0.003139653243124485, -0.0885978415608406, -0.0014193776296451688, -0.012879861518740654, 0.0005444222479127347, 0.009608214721083641, 0.024864858016371727, 0.0235707089304924, 0.07090839743614197, -0.0024098206777125597, -0.0026445190887898207, 0.2712360620498657, 0.08169154822826385, -0.003256028052419424, 0.0026594987139105797, 0.0333610400557518, 0.007371341343969107, -0.05751477926969528, -0.0766996294260025, -0.014951074495911598, 0.009272505529224873, 0.010496742092072964, -0.025111613795161247, -0.003911581356078386, -0.0045793368481099606, 0.11310561746358871, -0.02881818264722824, -0.0814693495631218, 0.04869652912020683, 0.13595326244831085, -0.0016495505115017295, -0.08564484864473343, -0.0156776811927557];

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
