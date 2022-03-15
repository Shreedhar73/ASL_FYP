import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:tflite/tflite.dart';

class DetectPage extends StatefulWidget {
  bool isActive;
  DetectPage({Key? key, this.isActive = false}) : super(key: key);

  @override
  State<DetectPage> createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  // void testModel() async {
  //   var recoginitions = await Tflite.runModelOnImage(
  //     path: 'assets/model/testA.png',
  //     imageMean: 255,
  //     asynch: true,
  //   );
  //   print(recoginitions.toString());
  // }

  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool isCameraReady = false;
  String? res;
  String label = '';
  double percentage = 0.0;

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final firstCamera = cameras[0];
    _controller = CameraController(firstCamera, ResolutionPreset.ultraHigh);
    _initializeControllerFuture = _controller.initialize().then((_) async {
      if (!mounted) {
        return;
      }
      setState(() {
        isCameraReady = true;
      });

      res = await Tflite.loadModel(
        model: "assets/model/converted_model.tflite",
        labels: "assets/model/labels.txt",
      );
      _controller.startImageStream(
        (image) async {
          Tflite.runModelOnFrame(
            bytesList: image.planes.map((plane) {
              return plane.bytes;
            }).toList(),
            imageHeight: image.height,
            imageWidth: image.width,
            threshold: 0.5,
            numResults: 1,
            asynch: true,
          ).then((value) {
            value!.map((res) {});
            {
              setState(() {
                label = value.first['label'].toString();
                percentage = value.first['confidence'] * 100.toStringAsFixed(2);
              });
            }
          });
        },
      );
    });
  }

  @override
  void initState() {
    _initializeCamera();
    super.initState();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  bool show = false;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: const Text('Detection screen'),
          backgroundColor: Colors.transparent,
        ),
        body: Stack(
          children: [
            Column(
              children: [
                Flexible(
                  flex: 5,
                  child: (isCameraReady)
                      ? Hero(
                          tag: 'button',
                          child: SizedBox(
                              width: double.infinity,
                              child: CameraPreview(_controller)),
                        )
                      : Container(),
                ),
                Flexible(
                  flex: 2,
                  child: AnimatedContainer(
                    decoration: BoxDecoration(
                      color: whiteColor,
                      borderRadius: BorderRadius.circular(30),
                    ),
                    duration: const Duration(milliseconds: 200),
                    width: displayWidth(context),
                    child: Column(
                      children: [
                        const SizedBox(
                          height: 20,
                        ),
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                'Position ASL signs in a camera to get the English equivalent',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  color: mainColor,
                                  fontFamily: 'Comfortaa',
                                  fontWeight: FontWeight.w800,
                                  fontSize: displayWidth(context) * 0.04,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(
                          height: 40,
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              label,
                              style: TextStyle(
                                color: mainColor,
                                fontFamily: 'Comfortaa',
                                fontWeight: FontWeight.w800,
                                fontSize: displayWidth(context) * 0.14,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const Text("Confidence level"),
                const Divider(),
                Text("   ${percentage.toString()}%"),
              ],
            ),
          ],
        ));
  }
}
