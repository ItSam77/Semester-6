import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FAB',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "ListView",
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: GridView.extent(
    maxCrossAxisExtent: 100,
    crossAxisSpacing: 10,
    mainAxisSpacing: 10,
    children: List.generate(50, (index) {
      return Padding(
        padding: const EdgeInsets.all(8.0),
        child: Container(
          color: Colors.green,
          child: Center(
            child: Text(
              "Extent: $index",
              style: TextStyle(color: Colors.white, fontSize: 10),
            ),
          ),
        ),
      );
    }),
  ),
    );
  }
}

