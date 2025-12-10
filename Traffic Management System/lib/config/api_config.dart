/// API Configuration for different environments
class ApiConfig {
  // Private constructor to prevent instantiation
  ApiConfig._();

  // Environment URLs
  static const String _devAndroidEmulator = 'http://10.0.2.2:5000';
  static const String _devIOSSimulator = 'http://127.0.0.1:5000';
  static const String _devLocalNetwork = 'http://192.168.1.100:5000'; // Change to your IP
  static const String _stagingUrl = 'https://staging-api.your-domain.com';
  static const String _productionUrl = 'https://api.your-domain.com';

  /// Current environment (set via --dart-define=ENV=prod)
  static const String environment = String.fromEnvironment(
    'ENV',
    defaultValue: 'dev',
  );

  /// Custom API URL override (set via --dart-define=API_URL=...)
  static const String? _customUrl = String.fromEnvironment('API_URL') != ''
      ? String.fromEnvironment('API_URL')
      : null;

  /// Get the base URL based on environment
  static String get baseUrl {
    // Custom URL takes priority
    if (_customUrl != null) {
      return _customUrl!;
    }

    switch (environment) {
      case 'prod':
      case 'production':
        return _productionUrl;
      case 'staging':
        return _stagingUrl;
      case 'dev':
      case 'development':
      default:
        // Default to Android emulator URL
        return _devAndroidEmulator;
    }
  }

  /// API Endpoints
  static Uri get predictEndpoint => Uri.parse('$baseUrl/api/v1/predict');
  static Uri get healthEndpoint => Uri.parse('$baseUrl/health');
  static Uri get readyEndpoint => Uri.parse('$baseUrl/ready');
  static Uri get trafficEndpoint => Uri.parse('$baseUrl/api/v1/traffic');
  static Uri get routesEndpoint => Uri.parse('$baseUrl/api/v1/routes');
  
  static Uri weatherEndpoint(String city) => 
      Uri.parse('$baseUrl/api/v1/weather/${Uri.encodeComponent(city)}');
  
  static Uri eventsEndpoint(String city) => 
      Uri.parse('$baseUrl/api/v1/events/${Uri.encodeComponent(city)}');

  /// Request timeout duration
  static const Duration requestTimeout = Duration(seconds: 30);
  
  /// Connection timeout duration
  static const Duration connectionTimeout = Duration(seconds: 10);

  /// API Headers
  static Map<String, String> get defaultHeaders => {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

  /// Debug info
  static void printConfig() {
    print('=== API Configuration ===');
    print('Environment: $environment');
    print('Base URL: $baseUrl');
    print('========================');
  }
}