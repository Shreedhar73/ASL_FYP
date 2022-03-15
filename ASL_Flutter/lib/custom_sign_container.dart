import 'package:aslflutter/utils/colors.dart';
import 'package:aslflutter/utils/sizes.dart';
import 'package:flutter/material.dart';

Widget SignContainer(
    {required BuildContext context,
    required String image,
    required String value}) {
  return SizedBox(
    height: displayWidth(context) * 0.4,
    width: displayWidth(context) * 0.3,
    child: Column(
      children: [
        Image.asset(
          image,
          width: displayWidth(context) * 0.2,
        ),
        Text(
          value,
          style: TextStyle(
            fontFamily: 'Comfortaa',
            fontWeight: FontWeight.w600,
            fontSize: displayWidth(context) * 0.12,
            color: mainColor,
          ),
        ),
      ],
    ),
  );
}
