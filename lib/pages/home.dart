import 'dart:async';
import 'dart:developer';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:face_auth/pages/face_image.dart';
import 'package:face_auth/pages/user_modal.dart';
import 'package:face_auth/utils/face_predictor.dart';
import 'package:face_auth/utils/firebase_utils.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

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

  bool isPredicting = false;

  List<UserModal> users = [];

  UserModal? matchedUser;

  String nameInput = "";

  bool isUsersDownloading = false;
  bool isNewUserPushing = false;
  bool isProcessing = false;

  initFaceDetector() {
    faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableLandmarks: true,
        performanceMode: FaceDetectorMode.accurate,
      ),
    );
  }

  InputImage? convertToInputImage(CameraImage image) {
    InputImageRotation? rotation = InputImageRotationValue.fromRawValue(0);

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

    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      log("Invalid image format !!!!!!!!!!!");
      return null;
    }

    // since format is constraint to nv21 or bgra8888, both only have one plane
    if (image.planes.length != 1) {
      log("So many planes !!!!!!!!!!!");
      return null;
    }

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

  Future<void> startPredictions() async {
    setState(() {
      isPredicting = true;
      matchedUser = null;
    });

    facePredictor.generateCurrentFacePredictionData();

    double maxSim = -1;

    for (var user in users) {
      double currSim = facePredictor.checkSimilarityWith(user.faceData!);

      if (currSim > 0.7 && currSim > maxSim) {
        matchedUser = user;
        maxSim = currSim;
      }
    }

    setState(() {
      isPredicting = false;
    });
  }

  Future<bool> processImage() async {
    if (currentImage == null) {
      log("Current Image is NULL !!!!!!!!!!!!");
      return false;
    }

    setState(() {
      isProcessing = true;
    });

    InputImage? inputImage = convertToInputImage(currentImage!);

    if (inputImage == null) return false;

    final List<Face> faces = await faceDetector.processImage(inputImage);

    log("Faces detected: ${faces.length}");

    bool b = false;

    if (faces.length != 1) {
      showDialog(
        context: context,
        builder: (context) {
          return const AlertDialog(
            title: Text("Try Again"),
            content: Text(
              "Your face not detected correctly.",
            ),
          );
        },
      );
    } else {
      facePredictor.captureFace(currentImage!, faces[0]);
      startPredictions();
      b = true;
    }

    setState(() {
      isProcessing = false;
    });

    return b;
  }

  startStream() {
    cameraController.startImageStream((CameraImage image) {
      currentImage = image;
    });
  }

  loadUsers() async {
    setState(() {
      isUsersDownloading = true;
    });

    await downloadUsers(users);

    setState(() {
      isUsersDownloading = false;
    });
  }

  addNewUser() async {
    setState(() {
      isNewUserPushing = true;
    });

    UserModal user = UserModal();

    user.setName(nameInput);
    user.setFaceData(facePredictor.predictedData);

    if (await pushNewUser(user)) {
      users.add(user);
      reset();
      return;
    }

    showDialog(
      context: context,
      builder: (context) {
        return const AlertDialog(
          title: Text("Save Failed!"),
          content: Text(
            "Your face unable to save in database. Please try again.",
          ),
        );
      },
    );

    setState(() {
      isNewUserPushing = false;
    });
  }

  reset() {
    isNewUserPushing = false;
    facePredictor.reset();
    setState(() {});
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
          : ImageFormatGroup.bgra8888,
    );

    cameraController.initialize().then((_) {
      if (!mounted) {
        log("Not mounted...!!!!!!!!!!!!!");
        return;
      }

      log("Camera ready..........ok");

      startStream();
      loadUsers();

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
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text("Face Auth"),
      ),
      body: facePredictor.currentFaceImage == null
          ? Container(
              color: Colors.black,
              height: double.infinity,
              width: double.infinity,
              child: !cameraController.value.isInitialized ||
                      !facePredictor.initialized
                  ? const Text("Not initialized yet")
                  : Stack(
                      children: [
                        Align(
                          alignment: Alignment.center,
                          child: CameraPreview(cameraController),
                        ),
                        Align(
                          alignment: Alignment.bottomCenter,
                          child: Container(
                            margin: const EdgeInsets.only(bottom: 20),
                            child: FilledButton(
                              onPressed: isProcessing || isUsersDownloading
                                  ? null
                                  : () async {
                                      if (await processImage()) setState(() {});
                                    },
                              child: const Text("PROCESS"),
                            ),
                          ),
                        ),
                        Align(
                          alignment: Alignment.topCenter,
                          child: Container(
                            color: Colors.white,
                            padding: const EdgeInsets.all(10),
                            child: Row(
                              children: [
                                !isUsersDownloading
                                    ? Text("${users.length} faces loaded")
                                    : const CircularProgressIndicator(),
                                const Spacer(),
                                OutlinedButton(
                                  onPressed: isUsersDownloading ? null : () {
                                    loadUsers();
                                  },
                                  child: const Text("RELOAD"),
                                ),
                              ],
                            ),
                          ),
                        ),
                        if (isProcessing)
                          Container(
                            width: double.infinity,
                            height: double.infinity,
                            padding: const EdgeInsets.all(20),
                            color: Colors.white.withOpacity(0.5),
                            child: const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    "Processing...",
                                    style: TextStyle(fontSize: 25),
                                  ),
                                  LinearProgressIndicator(),
                                ],
                              ),
                            ),
                          ),
                      ],
                    ))
          : Container(
              width: double.infinity,
              height: double.infinity,
              padding: const EdgeInsets.all(15),
              child: Column(
                children: [
                  FaceImage(image: facePredictor.currentFaceImage!),
                  const SizedBox(height: 20),
                  isPredicting
                      ? const LinearProgressIndicator()
                      : Column(
                          children: [
                            matchedUser != null
                                ? Column(
                                    children: [
                                      const Text("Hello,"),
                                      Text(
                                        matchedUser!.name!,
                                        style: const TextStyle(fontSize: 30),
                                      ),
                                    ],
                                  )
                                : Column(
                                    children: [
                                      const Text(
                                        "New face!",
                                        textAlign: TextAlign.center,
                                        style: TextStyle(fontSize: 30),
                                      ),
                                      const Text(
                                        "you can save your face now.",
                                        textAlign: TextAlign.center,
                                      ),
                                      const SizedBox(height: 20),
                                      TextField(
                                        onChanged: (value) {
                                          setState(() {
                                            nameInput = value.trim();
                                          });
                                        },
                                        decoration: const InputDecoration(
                                            hintText:
                                                "Enter your name here..."),
                                      ),
                                      const SizedBox(height: 20),
                                    ],
                                  ),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                OutlinedButton(
                                  onPressed: () {
                                    reset();
                                  },
                                  child: Text("BACK"),
                                ),
                                const SizedBox(width: 10),
                                if (matchedUser == null)
                                  Expanded(
                                    child: FilledButton(
                                      onPressed:
                                          nameInput.isEmpty || isNewUserPushing
                                              ? null
                                              : () {
                                                  addNewUser();
                                                },
                                      child: isNewUserPushing
                                          ? const CircularProgressIndicator()
                                          : const Text("SAVE"),
                                    ),
                                  ),
                              ],
                            ),
                          ],
                        ),
                ],
              ),
            ),
    );
  }
}
