import 'package:flutter/foundation.dart';

import '../models/traffic_prediction.dart';
import '../services/traffic_service.dart';

/// Loading state enumeration
enum LoadingState {
  initial,
  loading,
  loaded,
  error,
}

/// Provider for managing traffic prediction state
class TrafficProvider with ChangeNotifier {
  final TrafficService _service;

  LoadingState _state = LoadingState.initial;
  TrafficPredictionResult? _prediction;
  String? _error;
  bool _isApiAvailable = true;

  // Input state
  String _origin = '';
  String _destination = '';
  String _city = '';

  TrafficProvider(this._service);

  // Getters
  LoadingState get state => _state;
  TrafficPredictionResult? get prediction => _prediction;
  String? get error => _error;
  bool get isApiAvailable => _isApiAvailable;
  String get origin => _origin;
  String get destination => _destination;
  String get city => _city;

  // Computed getters
  bool get isLoading => _state == LoadingState.loading;
  bool get hasError => _state == LoadingState.error;
  bool get hasData => _state == LoadingState.loaded && _prediction != null;

  /// Update origin
  void setOrigin(String value) {
    _origin = value;
    notifyListeners();
  }

  /// Update destination
  void setDestination(String value) {
    _destination = value;
    notifyListeners();
  }

  /// Update city
  void setCity(String value) {
    _city = value;
    notifyListeners();
  }

  /// Check API availability
  Future<void> checkApiAvailability() async {
    try {
      _isApiAvailable = await _service.isApiAvailable();
      notifyListeners();
    } catch (e) {
      _isApiAvailable = false;
      notifyListeners();
    }
  }

  /// Fetch traffic prediction
  Future<void> fetchPrediction(
    String origin,
    String destination, {
    String? city,
  }) async {
    // Update input state
    _origin = origin;
    _destination = destination;
    _city = city ?? _extractCity(origin);

    // Set loading state
    _state = LoadingState.loading;
    _error = null;
    notifyListeners();

    try {
      _prediction = await _service.getPrediction(
        origin,
        destination,
        city: _city,
      );
      _state = LoadingState.loaded;
      _isApiAvailable = true;
    } on TrafficServiceException catch (e) {
      _error = e.message;
      _state = LoadingState.error;
      _isApiAvailable = !e.isNetworkError;
    } catch (e) {
      _error = 'An unexpected error occurred';
      _state = LoadingState.error;
    }

    notifyListeners();
  }

  /// Retry the last prediction
  Future<void> retry() async {
    if (_origin.isNotEmpty && _destination.isNotEmpty) {
      await fetchPrediction(_origin, _destination, city: _city);
    }
  }

  /// Reset to initial state
  void reset() {
    _state = LoadingState.initial;
    _prediction = null;
    _error = null;
    _origin = '';
    _destination = '';
    _city = '';
    notifyListeners();
  }

  /// Clear just the results (keep inputs)
  void clearResults() {
    _state = LoadingState.initial;
    _prediction = null;
    _error = null;
    notifyListeners();
  }

  /// Extract city from location string
  String _extractCity(String location) {
    // Simple extraction - take first part before comma
    final parts = location.split(',');
    return parts.first.trim();
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }
}