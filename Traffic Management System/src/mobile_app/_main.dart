// main.dart
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(TrafficApp());

class TrafficApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Traffic App')),
        body: TrafficPrediction(),
      ),
    );
  }
}

class TrafficPrediction extends StatefulWidget {
  @override
  _TrafficPredictionState createState() => _TrafficPredictionState();
}

class _TrafficPredictionState extends State<TrafficPrediction> {
  String _prediction = "Loading...";

  void _fetchPrediction() async {
    final response = await http.post(
      Uri.parse('http://127.0.0.1:5000/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({"temp": 25, "rain_1h": 0, "snow_1h": 0, "clouds_all": 50}),
    );
    if (response.statusCode == 200) {
      setState(() {
        _prediction = jsonDecode(response.body)['predicted_traffic_volume'].toString();
      });
    } else {
      setState(() {
        _prediction = "Error fetching prediction";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text("Predicted Traffic Volume: $_prediction"),
          ElevatedButton(
            onPressed: _fetchPrediction,
            child: Text("Get Prediction"),
          ),
        ],
      ),
    );
  }
}