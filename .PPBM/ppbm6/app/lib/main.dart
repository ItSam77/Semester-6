import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  static const header = 'GeeksforGeeks';

  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: header,
      home: MyHomePage(title: header),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatelessWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: const snackBarDemo(),
    );
  }
}

// ignore: camel_case_types
class snackBarDemo extends StatelessWidget {
  const snackBarDemo({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
            backgroundColor: Colors.green, foregroundColor: Colors.white),
        onPressed: () {
          const snackdemo = SnackBar(
            content: Text('Hii this is GFG\'s SnackBar'),
            backgroundColor: Colors.green,
            elevation: 10,
            behavior: SnackBarBehavior.floating,
            margin: EdgeInsets.all(5),
          );
          ScaffoldMessenger.of(context).showSnackBar(snackdemo);
        },
        child: const Text(
          'Click Here',
        ),
      ),
    );
  }
}