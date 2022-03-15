import 'package:aslflutter/Detect.dart';
import 'package:aslflutter/sign_learn_screen.dart';
import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
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
  int _selectedIndex = 0;

  static const List<Widget> _widgetOptions = <Widget>[
    Text(
      'Index 0: Home',
    ),
    Text(
      'Index 1: Business',
    ),
  ];
  _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
      print(_selectedIndex);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.search), label: 'Detect'),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.black,
        onTap: _onItemTapped,
      ),
      body: SafeArea(
          child: IndexedStack(
        index: _selectedIndex,
        children: <Widget>[
          SignLearnScreen(
            isActive: _selectedIndex == 1,
          ),
          DetectPage(isActive: false),
        ],
      )
          // child: ElevatedButton(
          //   child: Center(
          //     child: Row(
          //       mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          //       children: [
          //         Text(
          //           'Live Translation',
          //           style: TextStyle(
          //             fontFamily: 'Comfortaa',
          //             fontWeight: FontWeight.w800,
          //             fontSize: displayWidth(context) * 0.04,
          //             color: secondaryColor,
          //           ),
          //         ),
          //       ],
          //     ),
          //   ),
          //   style: ButtonStyle(
          //     backgroundColor: MaterialStateProperty.all(mainColor),
          //     foregroundColor: MaterialStateProperty.all(mainColor),
          //   ),
          //   onPressed: () {
          //     Get.to(const DetectPage());
          //   },
          // ),
          ),
    );
  }
}
