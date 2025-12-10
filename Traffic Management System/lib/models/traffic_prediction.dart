/// Main traffic prediction result from the API
class TrafficPredictionResult {
  final String origin;
  final String destination;
  final String city;
  final String overallStatus;
  final String overallSeverity;
  final String summary;
  final List<String> recommendations;
  final TrafficData? traffic;
  final WeatherData? weather;
  final EventData? events;
  final RoutingData? routing;
  final int successfulAgents;
  final List<String> failedAgents;
  final double executionTimeMs;
  final String timestamp;

  TrafficPredictionResult({
    required this.origin,
    required this.destination,
    required this.city,
    required this.overallStatus,
    required this.overallSeverity,
    required this.summary,
    required this.recommendations,
    this.traffic,
    this.weather,
    this.events,
    this.routing,
    required this.successfulAgents,
    required this.failedAgents,
    required this.executionTimeMs,
    required this.timestamp,
  });

  factory TrafficPredictionResult.fromJson(Map<String, dynamic> json) {
    final agents = json['agents'] as Map<String, dynamic>? ?? {};

    return TrafficPredictionResult(
      origin: json['origin'] ?? '',
      destination: json['destination'] ?? '',
      city: json['city'] ?? '',
      overallStatus: json['overall_status'] ?? 'unknown',
      overallSeverity: json['overall_severity'] ?? 'unknown',
      summary: json['summary'] ?? '',
      recommendations: List<String>.from(json['recommendations'] ?? []),
      traffic: _parseAgentData<TrafficData>(
        agents['traffic'],
        TrafficData.fromJson,
      ),
      weather: _parseAgentData<WeatherData>(
        agents['weather'],
        WeatherData.fromJson,
      ),
      events: _parseAgentData<EventData>(
        agents['events'],
        EventData.fromJson,
      ),
      routing: _parseAgentData<RoutingData>(
        agents['routing'],
        RoutingData.fromJson,
      ),
      successfulAgents: json['successful_agents'] ?? 0,
      failedAgents: List<String>.from(json['failed_agents'] ?? []),
      executionTimeMs: (json['execution_time_ms'] ?? 0).toDouble(),
      timestamp: json['timestamp'] ?? DateTime.now().toIso8601String(),
    );
  }

  static T? _parseAgentData<T>(
    Map<String, dynamic>? agentResult,
    T Function(Map<String, dynamic>) fromJson,
  ) {
    if (agentResult == null) return null;
    if (agentResult['status'] != 'success') return null;
    if (agentResult['data'] == null) return null;
    
    try {
      return fromJson(agentResult['data'] as Map<String, dynamic>);
    } catch (e) {
      return null;
    }
  }

  /// Check if this is a successful prediction
  bool get isSuccess => overallStatus == 'success';

  /// Check if there are any warnings
  bool get hasWarnings => 
      overallSeverity == 'medium' || overallSeverity == 'high';

  /// Get severity level as enum
  SeverityLevel get severityLevel {
    switch (overallSeverity) {
      case 'high':
        return SeverityLevel.high;
      case 'medium':
        return SeverityLevel.medium;
      case 'low':
        return SeverityLevel.low;
      default:
        return SeverityLevel.unknown;
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'origin': origin,
      'destination': destination,
      'city': city,
      'overall_status': overallStatus,
      'overall_severity': overallSeverity,
      'summary': summary,
      'recommendations': recommendations,
      'successful_agents': successfulAgents,
      'failed_agents': failedAgents,
      'execution_time_ms': executionTimeMs,
      'timestamp': timestamp,
    };
  }
}

/// Severity level enumeration
enum SeverityLevel {
  low,
  medium,
  high,
  unknown,
}

/// Traffic analysis data
class TrafficData {
  final String congestionLevel;
  final double congestionScore;
  final bool isCongested;
  final String recommendedAction;
  final int estimatedDelayMinutes;
  final String? signalTimingAdjustment;
  final String confidence;
  final List<String> contributingFactors;

  TrafficData({
    required this.congestionLevel,
    required this.congestionScore,
    required this.isCongested,
    required this.recommendedAction,
    required this.estimatedDelayMinutes,
    this.signalTimingAdjustment,
    required this.confidence,
    required this.contributingFactors,
  });

  factory TrafficData.fromJson(Map<String, dynamic> json) {
    return TrafficData(
      congestionLevel: json['congestion_level'] ?? 'unknown',
      congestionScore: (json['congestion_score'] ?? 0).toDouble(),
      isCongested: json['is_congested'] ?? false,
      recommendedAction: json['recommended_action'] ?? '',
      estimatedDelayMinutes: json['estimated_delay_minutes'] ?? 0,
      signalTimingAdjustment: json['signal_timing_adjustment'],
      confidence: json['confidence'] ?? 'unknown',
      contributingFactors: List<String>.from(json['contributing_factors'] ?? []),
    );
  }
}

/// Weather analysis data
class WeatherData {
  final String condition;
  final int conditionCode;
  final String description;
  final double temperature;
  final double feelsLike;
  final int humidity;
  final double windSpeed;
  final double? windGust;
  final int visibility;
  final String visibilityCategory;
  final double precipitation1h;
  final double snow1h;
  final String severity;
  final bool isHazardous;
  final bool hasPrecipitation;
  final List<String> recommendations;
  final int trafficSpeedReductionPct;

  WeatherData({
    required this.condition,
    required this.conditionCode,
    required this.description,
    required this.temperature,
    required this.feelsLike,
    required this.humidity,
    required this.windSpeed,
    this.windGust,
    required this.visibility,
    required this.visibilityCategory,
    required this.precipitation1h,
    required this.snow1h,
    required this.severity,
    required this.isHazardous,
    required this.hasPrecipitation,
    required this.recommendations,
    required this.trafficSpeedReductionPct,
  });

  factory WeatherData.fromJson(Map<String, dynamic> json) {
    return WeatherData(
      condition: json['condition'] ?? 'Unknown',
      conditionCode: json['condition_code'] ?? 0,
      description: json['description'] ?? '',
      temperature: (json['temperature'] ?? 0).toDouble(),
      feelsLike: (json['feels_like'] ?? 0).toDouble(),
      humidity: json['humidity'] ?? 0,
      windSpeed: (json['wind_speed'] ?? 0).toDouble(),
      windGust: json['wind_gust']?.toDouble(),
      visibility: json['visibility'] ?? 10000,
      visibilityCategory: json['visibility_category'] ?? 'good',
      precipitation1h: (json['precipitation_1h'] ?? 0).toDouble(),
      snow1h: (json['snow_1h'] ?? 0).toDouble(),
      severity: json['severity'] ?? 'clear',
      isHazardous: json['is_hazardous'] ?? false,
      hasPrecipitation: json['has_precipitation'] ?? false,
      recommendations: List<String>.from(json['recommendations'] ?? []),
      trafficSpeedReductionPct: json['traffic_speed_reduction_pct'] ?? 0,
    );
  }

  /// Get weather icon based on condition
  String get weatherIcon {
    final condition = this.condition.toLowerCase();
    if (condition.contains('clear')) return '☀️';
    if (condition.contains('cloud')) return '☁️';
    if (condition.contains('rain')) return '🌧️';
    if (condition.contains('thunder')) return '⛈️';
    if (condition.contains('snow')) return '❄️';
    if (condition.contains('fog') || condition.contains('mist')) return '🌫️';
    return '🌤️';
  }
}

/// Event analysis data
class EventData {
  final String level;
  final int eventCount;
  final int totalExpectedAttendance;
  final List<String> peakTimes;
  final String message;
  final List<String> recommendations;

  EventData({
    required this.level,
    required this.eventCount,
    required this.totalExpectedAttendance,
    required this.peakTimes,
    required this.message,
    required this.recommendations,
  });

  factory EventData.fromJson(Map<String, dynamic> json) {
    return EventData(
      level: json['level'] ?? 'low',
      eventCount: json['event_count'] ?? 0,
      totalExpectedAttendance: json['total_expected_attendance'] ?? 0,
      peakTimes: List<String>.from(json['peak_times'] ?? []),
      message: json['message'] ?? '',
      recommendations: List<String>.from(json['recommendations'] ?? []),
    );
  }

  bool get hasEvents => eventCount > 0;
}

/// Routing analysis data
class RoutingData {
  final List<RouteInfo> routes;
  final int recommendedIndex;
  final String recommendationReason;

  RoutingData({
    required this.routes,
    required this.recommendedIndex,
    required this.recommendationReason,
  });

  factory RoutingData.fromJson(Map<String, dynamic> json) {
    // Handle both list format and object format
    if (json is List) {
      return RoutingData(
        routes: (json).map((r) => RouteInfo.fromJson(r)).toList(),
        recommendedIndex: 0,
        recommendationReason: 'First available route',
      );
    }

    final routesList = json['routes'] ?? json['data'] ?? [];
    
    return RoutingData(
      routes: (routesList as List).map((r) => RouteInfo.fromJson(r)).toList(),
      recommendedIndex: json['recommended_index'] ?? 0,
      recommendationReason: json['recommendation_reason'] ?? '',
    );
  }

  RouteInfo? get recommendedRoute {
    if (routes.isEmpty) return null;
    if (recommendedIndex < 0 || recommendedIndex >= routes.length) {
      return routes.first;
    }
    return routes[recommendedIndex];
  }
}

/// Individual route information
class RouteInfo {
  final int routeIndex;
  final String summary;
  final String distanceText;
  final int distanceMeters;
  final String durationText;
  final int durationSeconds;
  final String? durationInTrafficText;
  final int? durationInTrafficSeconds;
  final String trafficCondition;
  final List<String> warnings;

  RouteInfo({
    required this.routeIndex,
    required this.summary,
    required this.distanceText,
    required this.distanceMeters,
    required this.durationText,
    required this.durationSeconds,
    this.durationInTrafficText,
    this.durationInTrafficSeconds,
    required this.trafficCondition,
    required this.warnings,
  });

  factory RouteInfo.fromJson(Map<String, dynamic> json) {
    return RouteInfo(
      routeIndex: json['route_index'] ?? 0,
      summary: json['summary'] ?? 'Route',
      distanceText: json['distance_text'] ?? 'N/A',
      distanceMeters: json['distance_meters'] ?? 0,
      durationText: json['duration_text'] ?? 'N/A',
      durationSeconds: json['duration_seconds'] ?? 0,
      durationInTrafficText: json['duration_in_traffic_text'],
      durationInTrafficSeconds: json['duration_in_traffic_seconds'],
      trafficCondition: json['traffic_condition'] ?? 'unknown',
      warnings: List<String>.from(json['warnings'] ?? []),
    );
  }

  /// Calculate delay in minutes
  int get delayMinutes {
    if (durationInTrafficSeconds == null) return 0;
    return ((durationInTrafficSeconds! - durationSeconds) / 60).round();
  }
}