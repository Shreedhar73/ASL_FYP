import 'package:flutter/material.dart';
import 'package:imageclassification/utils/colors.dart';
import 'package:imageclassification/utils/sizes.dart';

Widget SignContainer(
    {required BuildContext context,
    required String image,
    required String value}) {
  return SizedBox(
    height: displayWidth(context) * 0.5,
    width: displayWidth(context) * 0.3,
    child: Column(
      mainAxisAlignment: MainAxisAlignment.spaceAround,
      children: [
        Expanded(
          child: Image.asset(
            image,
            width: displayWidth(context) * 0.2,
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: TextStyle(
              fontFamily: 'Comfortaa',
              fontWeight: FontWeight.w600,
              fontSize: 40,
              color: mainColor,
            ),
          ),
        ),
      ],
    ),
  );
}
