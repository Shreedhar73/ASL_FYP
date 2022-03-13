import 'package:aslflutter/Detect.dart';
import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_core/get_core.dart';
import 'package:tflite/tflite.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

String? res;
void loadModel() async {
  res = await Tflite.loadModel(
    model: "assets/model/phone_VGG16--89--08-03-01-46.tflite",
    labels: "assets/model/labels_mobilenet_quant_v1_224.txt",
  );
}

class _HomePageState extends State<HomePage> {
  void testModel() async {
    var recoginitions = await Tflite.runModelOnImage(
      path: 'assets/model/testA.png',
      imageMean: 255,
      asynch: true,
    );
  }

  @override
  void initState() {
    // loadModel();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton(
          child: Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Text(
                  'Live Translation',
                  style: TextStyle(
                    fontFamily: 'Comfortaa',
                    fontWeight: FontWeight.w800,
                    fontSize: displayWidth(context) * 0.04,
                    color: secondaryColor,
                  ),
                ),
              ],
            ),
          ),
          style: ButtonStyle(
            backgroundColor: MaterialStateProperty.all(mainColor),
            foregroundColor: MaterialStateProperty.all(mainColor),
          ),
          // splashColor: mainColor,
          onPressed: () {
            Get.to(const DetectPage());
          },
          // color: mainColor,
        ),
      ),
    );
  }
}