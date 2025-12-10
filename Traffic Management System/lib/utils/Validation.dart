/// Validation utilities for the app

/// Validate location input
String? validateLocation(String? value, {String fieldName = 'Location'}) {
  if (value == null || value.trim().isEmpty) {
    return '$fieldName is required';
  }

  final trimmed = value.trim();

  if (trimmed.length < 2) {
    return '$fieldName must be at least 2 characters';
  }

  if (trimmed.length > 200) {
    return '$fieldName is too long (max 200 characters)';
  }

  // Check for invalid characters
  final invalidChars = RegExp(r'[<>{}|\\^`\x00-\x1f]');
  if (invalidChars.hasMatch(trimmed)) {
    return '$fieldName contains invalid characters';
  }

  return null;
}

/// Validate origin
String? validateOrigin(String? value) {
  return validateLocation(value, fieldName: 'Origin');
}

/// Validate destination
String? validateDestination(String? value) {
  return validateLocation(value, fieldName: 'Destination');
}

/// Validate city
String? validateCity(String? value) {
  return validateLocation(value, fieldName: 'City');
}

/// Check if a string is a valid coordinate format
bool isCoordinateFormat(String value) {
  // Matches formats like "40.7128,-74.0060" or "40.7128, -74.0060"
  final coordRegex = RegExp(
    r'^-?\d+\.?\d*\s*,\s*-?\d+\.?\d*$',
  );
  return coordRegex.hasMatch(value.trim());
}

/// Extract city name from a location string
String extractCity(String location) {
  // Handle coordinate format
  if (isCoordinateFormat(location)) {
    return 'Unknown';
  }

  // Split by comma and take first part
  final parts = location.split(',');
  return parts.first.trim();
}

/// Sanitize location input
String sanitizeLocation(String value) {
  return value.trim().replaceAll(RegExp(r'\s+'), ' ');
}