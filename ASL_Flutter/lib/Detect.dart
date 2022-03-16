import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:tflite/tflite.dart';

class DetectPage extends StatefulWidget {
  @override
  _DetectPageState createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  void testModel() async {
    var recoginitions = await Tflite.runModelOnImage(
      path: 'assets/model/testA.png',
      imageMean: 255,
      asynch: true,
    );
    print(recoginitions.toString());
  }

  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool isCameraReady = false;

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

      String? res = await Tflite.loadModel(
        model: "assets/model/phone_VGG16--92--08-02-23-36.tflite",
        labels: "assets/model/labels.txt",
      );
      print('start');
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
            print(value!.first);
            //   print('yes');
            setState(() {
              label = value.first['label'].toString();
              percentage = value.first['confidence'].toString();
            });
            //   print(label);
          });
        },
      );
    });
  }

  String label = '';
  String percentage = "";
  @override
  void initState() {
    _initializeCamera();
    super.initState();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  bool show = false;

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
        systemNavigationBarColor: whiteColor,
      ),
      child: WillPopScope(
        onWillPop: () async {
          await _controller?.dispose()?.then(
                (context) => Get.back(),
              );
          return true;
        },
        child: Scaffold(
            appBar: AppBar(
              backgroundColor: secondaryColor,
              automaticallyImplyLeading: true,
              leading: Hero(
                tag: 'back',
                child: GestureDetector(
                  onTap: () {
                    _controller?.dispose()?.then(
                          (context) => Get.back(),
                        );
                  },
                  child: Icon(
                    Icons.arrow_back,
                    color: mainColor,
                  ),
                ),
              ),
            ),
            backgroundColor: whiteColor,
            body: Stack(
              children: [
                Column(
                  children: [
                    Flexible(
                      flex: 5,
                      child: (isCameraReady)
                          ? Hero(
                              tag: 'button',
                              child: Container(
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
                        duration: Duration(milliseconds: 200),
                        width: displayWidth(context),
                        child: Column(
                          children: [
                            SizedBox(
                              height: 20,
                            ),
                            Row(
                              children: [
                                Expanded(
                                  child: Text(
                                    'Position ASL signs in the viewfinder above to get the English equivalent below :',
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
                            SizedBox(
                              height: 40,
                            ),
                            Text(
                              label,
                              style: TextStyle(
                                color: mainColor,
                                fontFamily: 'Comfortaa',
                                fontWeight: FontWeight.w800,
                                fontSize: displayWidth(context) * 0.14,
                              ),
                            ),
                            Text(
                              percentage,
                              style: TextStyle(
                                fontSize: 10,
                              ),
                            )
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            )),
      ),
    );
  }
}
