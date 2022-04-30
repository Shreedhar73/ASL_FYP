import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:imageclassification/classifier.dart';
import 'package:imageclassification/classifier_float.dart';
import 'package:imageclassification/utils/colors.dart';
import 'package:logger/logger.dart';
import 'package:image/image.dart' as img;

import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class DetectPage extends StatefulWidget {
  DetectPage({Key? key, this.title}) : super(key: key);

  final String? title;

  @override
  _DetectPageState createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  late Classifier _classifier;
  var logger = Logger();
  File? _image;
  final picker = ImagePicker();
  Image? _imageWidget;
  img.Image? fox;
  Category? category;

  @override
  void initState() {
    super.initState();
    _classifier = ClassifierFloat();
  }

  Future getImage() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _imageWidget = Image.file(_image!);
        _predict();
      });
    }
  }

  void _predict() async {
    img.Image imageInput = img.decodeImage(_image!.readAsBytesSync())!;
    var pred = _classifier.predict(imageInput);

    if (pred.score < 0.9) {
      setState(() {
        this.category = Category("False Object", 0.0);
      });
    } else {
      setState(() {
        this.category = pred;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: secondaryColor,
      appBar: AppBar(
        backgroundColor: mainColor,
        title: Text('TfLite Flutter Helper',
            style: TextStyle(color: secondaryColor)),
      ),
      body: Column(
        children: <Widget>[
          Center(
            child: _image == null
                ? Text(
                    'No image selected.',
                    style: TextStyle(color: mainColor),
                  )
                : Container(
                    constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height / 2),
                    decoration: BoxDecoration(
                      border: Border.all(),
                    ),
                    child: _imageWidget,
                  ),
          ),
          SizedBox(
            height: 36,
          ),
          Text(
            category != null ? category!.label : '',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
          ),
          SizedBox(
            height: 8,
          ),
          Text(
            category != null
                ? 'Confidence: ${category!.score.toStringAsFixed(3)}'
                : '',
            style: TextStyle(fontSize: 16),
          ),
          SizedBox(
            height: 50,
          ),
          if (category != null)
            ElevatedButton(
              onPressed: () {
                setState(() {
                  _image = null;
                  category = null;
                });
              },
              child: Icon(Icons.close),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: mainColor,
        onPressed: getImage,
        tooltip: 'Pick Image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }
}
