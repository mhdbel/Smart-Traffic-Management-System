/// Generic API response wrapper
class ApiResponse<T> {
  final bool success;
  final T? data;
  final String? error;
  final int? statusCode;

  ApiResponse({
    required this.success,
    this.data,
    this.error,
    this.statusCode,
  });

  factory ApiResponse.success(T data) {
    return ApiResponse(success: true, data: data);
  }

  factory ApiResponse.failure(String error, {int? statusCode}) {
    return ApiResponse(
      success: false,
      error: error,
      statusCode: statusCode,
    );
  }
}

/// API error response structure
class ApiError {
  final String error;
  final String message;
  final int statusCode;

  ApiError({
    required this.error,
    required this.message,
    required this.statusCode,
  });

  factory ApiError.fromJson(Map<String, dynamic> json) {
    return ApiError(
      error: json['error'] ?? 'Unknown Error',
      message: json['message'] ?? 'An unknown error occurred',
      statusCode: json['status_code'] ?? 500,
    );
  }

  @override
  String toString() => message;
}

/// Health check response
class HealthResponse {
  final String status;
  final String service;
  final String version;
  final double? timestamp;

  HealthResponse({
    required this.status,
    required this.service,
    required this.version,
    this.timestamp,
  });

  factory HealthResponse.fromJson(Map<String, dynamic> json) {
    return HealthResponse(
      status: json['status'] ?? 'unknown',
      service: json['service'] ?? 'unknown',
      version: json['version'] ?? '0.0.0',
      timestamp: json['timestamp']?.toDouble(),
    );
  }

  bool get isHealthy => status == 'healthy';
}

/// Readiness check response
class ReadinessResponse {
  final String status;
  final Map<String, bool> checks;

  ReadinessResponse({
    required this.status,
    required this.checks,
  });

  factory ReadinessResponse.fromJson(Map<String, dynamic> json) {
    final checksMap = (json['checks'] as Map<String, dynamic>?) ?? {};
    
    return ReadinessResponse(
      status: json['status'] ?? 'unknown',
      checks: checksMap.map((key, value) => MapEntry(key, value == true)),
    );
  }

  bool get isReady => status == 'ready';
  
  List<String> get failedChecks => 
      checks.entries.where((e) => !e.value).map((e) => e.key).toList();
}