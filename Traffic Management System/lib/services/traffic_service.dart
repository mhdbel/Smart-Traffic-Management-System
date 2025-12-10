import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/traffic_prediction.dart';
import '../models/api_response.dart';

/// Service for interacting with the Traffic Management API
class TrafficService {
  final http.Client _client;
  
  TrafficService({http.Client? client}) : _client = client ?? http.Client();

  /// Get full traffic prediction
  Future<TrafficPredictionResult> getPrediction(
    String origin,
    String destination, {
    String? city,
  }) async {
    final response = await _makeRequest(
      method: 'POST',
      uri: ApiConfig.predictEndpoint,
      body: {
        'origin': origin,
        'destination': destination,
        if (city != null) 'city': city,
      },
    );

    return TrafficPredictionResult.fromJson(response);
  }

  /// Get traffic analysis only
  Future<TrafficData> getTrafficOnly(
    String origin,
    String destination,
  ) async {
    final response = await _makeRequest(
      method: 'POST',
      uri: ApiConfig.trafficEndpoint,
      body: {
        'origin': origin,
        'destination': destination,
      },
    );

    final trafficData = response['traffic']?['data'];
    if (trafficData == null) {
      throw TrafficServiceException('No traffic data available', 404);
    }

    return TrafficData.fromJson(trafficData);
  }

  /// Get weather for a city
  Future<WeatherData> getWeather(String city) async {
    final response = await _makeRequest(
      method: 'GET',
      uri: ApiConfig.weatherEndpoint(city),
    );

    final weatherData = response['weather']?['data'];
    if (weatherData == null) {
      throw TrafficServiceException('No weather data available', 404);
    }

    return WeatherData.fromJson(weatherData);
  }

  /// Get events for a city
  Future<EventData> getEvents(String city) async {
    final response = await _makeRequest(
      method: 'GET',
      uri: ApiConfig.eventsEndpoint(city),
    );

    final eventsData = response['events']?['data'];
    if (eventsData == null) {
      throw TrafficServiceException('No events data available', 404);
    }

    return EventData.fromJson(eventsData);
  }

  /// Get alternative routes
  Future<RoutingData> getRoutes(
    String origin,
    String destination,
  ) async {
    final response = await _makeRequest(
      method: 'POST',
      uri: ApiConfig.routesEndpoint,
      body: {
        'origin': origin,
        'destination': destination,
      },
    );

    final routesData = response['routes']?['data'];
    if (routesData == null) {
      throw TrafficServiceException('No route data available', 404);
    }

    return RoutingData.fromJson(routesData);
  }

  /// Check API health
  Future<HealthResponse> healthCheck() async {
    final response = await _makeRequest(
      method: 'GET',
      uri: ApiConfig.healthEndpoint,
    );

    return HealthResponse.fromJson(response);
  }

  /// Check API readiness
  Future<ReadinessResponse> readinessCheck() async {
    final response = await _makeRequest(
      method: 'GET',
      uri: ApiConfig.readyEndpoint,
    );

    return ReadinessResponse.fromJson(response);
  }

  /// Check if the API is reachable
  Future<bool> isApiAvailable() async {
    try {
      final health = await healthCheck();
      return health.isHealthy;
    } catch (e) {
      return false;
    }
  }

  /// Internal method to make HTTP requests
  Future<Map<String, dynamic>> _makeRequest({
    required String method,
    required Uri uri,
    Map<String, dynamic>? body,
  }) async {
    try {
      http.Response response;

      switch (method.toUpperCase()) {
        case 'GET':
          response = await _client
              .get(uri, headers: ApiConfig.defaultHeaders)
              .timeout(ApiConfig.requestTimeout);
          break;
        case 'POST':
          response = await _client
              .post(
                uri,
                headers: ApiConfig.defaultHeaders,
                body: jsonEncode(body),
              )
              .timeout(ApiConfig.requestTimeout);
          break;
        default:
          throw TrafficServiceException('Unsupported HTTP method: $method', 400);
      }

      return _handleResponse(response);
    } on SocketException {
      throw TrafficServiceException(
        'No internet connection. Please check your network.',
        0,
      );
    } on TimeoutException {
      throw TrafficServiceException(
        'Request timed out. Please try again.',
        408,
      );
    } on FormatException {
      throw TrafficServiceException(
        'Invalid response format from server.',
        500,
      );
    } on TrafficServiceException {
      rethrow;
    } catch (e) {
      throw TrafficServiceException(
        'An unexpected error occurred: ${e.toString()}',
        500,
      );
    }
  }

  /// Handle HTTP response
  Map<String, dynamic> _handleResponse(http.Response response) {
    final body = response.body;
    
    Map<String, dynamic> data;
    try {
      data = jsonDecode(body) as Map<String, dynamic>;
    } catch (e) {
      throw TrafficServiceException(
        'Invalid response from server',
        response.statusCode,
      );
    }

    if (response.statusCode >= 200 && response.statusCode < 300) {
      return data;
    }

    // Handle error responses
    final error = ApiError.fromJson(data);
    throw TrafficServiceException(error.message, response.statusCode);
  }

  /// Dispose of resources
  void dispose() {
    _client.close();
  }
}

/// Custom exception for traffic service errors
class TrafficServiceException implements Exception {
  final String message;
  final int statusCode;

  TrafficServiceException(this.message, this.statusCode);

  @override
  String toString() => message;

  /// Check if this is a network error
  bool get isNetworkError => statusCode == 0;

  /// Check if this is a client error (4xx)
  bool get isClientError => statusCode >= 400 && statusCode < 500;

  /// Check if this is a server error (5xx)
  bool get isServerError => statusCode >= 500;

  /// Check if this is a timeout
  bool get isTimeout => statusCode == 408;
}