import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:flutter/material.dart';

Widget SignContainer(
    {required BuildContext context,
    required String image,
    required String value}) {
  return SizedBox(
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
