import 'package:aslflutter/Detect.dart';
import 'package:aslflutter/custom_sign_container.dart';
import 'package:aslflutter/sign_map.dart';
import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:flutter/material.dart';
import 'package:sliding_up_panel/sliding_up_panel.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool panelOpen = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: secondaryColor,
      body: SafeArea(
        child: SlidingUpPanel(
          onPanelOpened: () {
            setState(() {
              panelOpen = true;
            });
          },
          onPanelClosed: () {
            setState(() {
              panelOpen = false;
            });
          },
          minHeight: displayHeight(context) * 0.45,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(60),
          ),
          panel: Container(
            decoration: BoxDecoration(
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(60),
              ),
              color: whiteColor,
            ),
            child: Padding(
              padding: const EdgeInsets.only(
                top: 20.0,
                left: 10,
                right: 10,
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: displayWidth(context) * 0.12,
                        height: 3,
                        decoration: BoxDecoration(
                          color: secondaryColor,
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(
                    height: 15,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Dictionary',
                        style: TextStyle(
                          color: secondaryColor,
                          fontWeight: FontWeight.w800,
                          fontFamily: 'Comfortaa',
                          fontSize: displayWidth(context) * 0.06,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(
                    height: 5,
                  ),
                  Expanded(
                    child: Center(
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        shrinkWrap: true,
                        itemCount: handSignMap.length,
                        physics: (panelOpen)
                            ? const BouncingScrollPhysics()
                            : const NeverScrollableScrollPhysics(),
                        itemBuilder: (BuildContext context, int index) {
                          return SignContainer(
                              context: context,
                              image: handSignMap.keys.toList()[index],
                              value: handSignMap.values.toList()[index]);
                        },
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          body: Padding(
            padding: const EdgeInsets.all(30.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                const SizedBox(
                  height: 10,
                ),
                Row(
                  children: [
                    Text(
                      'A2E',
                      style: TextStyle(
                        fontFamily: 'Comfortaa',
                        fontWeight: FontWeight.w600,
                        fontSize: displayWidth(context) * 0.1,
                        color: mainColor,
                      ),
                    ),
                  ],
                ),
                const SizedBox(
                  height: 5,
                ),
                Row(
                  children: [
                    SizedBox(
                      width: displayWidth(context) * 0.7,
                      child: Text(
                        'Because communication should have no boundaries',
                        style: TextStyle(
                          fontFamily: 'Comfortaa',
                          fontWeight: FontWeight.w800,
                          fontSize: displayWidth(context) * 0.04,
                          color: mainColor,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(
                  height: 30,
                ),
                Row(
                  children: [
                    Hero(
                      tag: 'button',
                      child: Container(
                        width: displayWidth(context) * 0.5,
                        height: displayWidth(context) * 0.14,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(20),
                          boxShadow: const <BoxShadow>[
                            BoxShadow(
                              color: Colors.black26,
                              offset: Offset(0, 10),
                              spreadRadius: -5,
                              blurRadius: 10,
                            ),
                          ],
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: ElevatedButton(
                            child: Center(
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceEvenly,
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
                              backgroundColor:
                                  MaterialStateProperty.all(mainColor),
                              foregroundColor:
                                  MaterialStateProperty.all(mainColor),
                            ),
                            onPressed: () {
                              Navigator.push(
                                context,
                                SlideRightRoute(page: DetectPage()),
                              );
                            },
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(
                      width: 10,
                    ),
                    Hero(
                      tag: 'back',
                      child: Icon(
                        Icons.arrow_forward,
                        color: mainColor,
                        size: displayWidth(context) * 0.07,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class SlideRightRoute extends PageRouteBuilder {
  final Widget? page;
  SlideRightRoute({this.page})
      : super(
          pageBuilder: (
            BuildContext context,
            Animation<double> animation,
            Animation<double> secondaryAnimation,
          ) =>
              page!,
          transitionsBuilder: (
            BuildContext context,
            Animation<double> animation,
            Animation<double> secondaryAnimation,
            Widget child,
          ) =>
              SlideTransition(
            position: Tween<Offset>(
              begin: const Offset(1, 0),
              end: Offset.zero,
            ).animate(animation),
            child: child,
          ),
        );
}
