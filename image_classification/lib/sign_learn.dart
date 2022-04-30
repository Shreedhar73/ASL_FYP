import 'package:flutter/material.dart';
import 'package:imageclassification/sign_map.dart';

class SignLearnScreen extends StatelessWidget {
  SignLearnScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text('Learn Sign'),
        backgroundColor: Colors.transparent,
      ),
      body: SafeArea(
        child: ListView.separated(
          padding: const EdgeInsets.all(8),
          itemCount: handSignMap.length,
          itemBuilder: (BuildContext context, int index) {
            return SizedBox(
              height: 50,
              child: ListTile(
                leading: CircleAvatar(
                  radius: 25,
                  backgroundColor: Colors.white,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Image.asset(handSignMap.keys.toList()[index]),
                    ),
                  ),
                ),
                trailing: Text(handSignMap.values.toList()[index]),
                title: const Center(
                  child: Text("=>"),
                ),
              ),
            );
          },
          separatorBuilder: (BuildContext context, int index) =>
              const Divider(),
        ),
      ),
    );
  }
}
