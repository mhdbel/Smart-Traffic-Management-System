import 'package:flutter/material.dart';

/// Reusable location input widget with validation
class LocationInput extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final String hint;
  final IconData icon;
  final Color? iconColor;
  final TextInputAction? textInputAction;
  final VoidCallback? onSubmitted;
  final String? Function(String?)? validator;
  final bool enabled;
  final bool autofocus;

  const LocationInput({
    super.key,
    required this.controller,
    required this.label,
    required this.hint,
    required this.icon,
    this.iconColor,
    this.textInputAction,
    this.onSubmitted,
    this.validator,
    this.enabled = true,
    this.autofocus = false,
  });

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller: controller,
      enabled: enabled,
      autofocus: autofocus,
      textInputAction: textInputAction ?? TextInputAction.next,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon: Icon(icon, color: iconColor),
        suffixIcon: controller.text.isNotEmpty
            ? IconButton(
                icon: const Icon(Icons.clear, size: 20),
                onPressed: () {
                  controller.clear();
                },
              )
            : null,
      ),
      validator: validator ?? _defaultValidator,
      onFieldSubmitted: (_) => onSubmitted?.call(),
    );
  }

  String? _defaultValidator(String? value) {
    if (value == null || value.trim().isEmpty) {
      return '$label is required';
    }
    if (value.trim().length < 2) {
      return '$label must be at least 2 characters';
    }
    if (value.trim().length > 200) {
      return '$label is too long (max 200 characters)';
    }
    return null;
  }
}

/// Origin input widget
class OriginInput extends StatelessWidget {
  final TextEditingController controller;
  final TextInputAction? textInputAction;
  final VoidCallback? onSubmitted;
  final bool enabled;

  const OriginInput({
    super.key,
    required this.controller,
    this.textInputAction,
    this.onSubmitted,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return LocationInput(
      controller: controller,
      label: 'Origin',
      hint: 'e.g., Rabat, Morocco',
      icon: Icons.trip_origin,
      iconColor: Colors.green,
      textInputAction: textInputAction,
      onSubmitted: onSubmitted,
      enabled: enabled,
    );
  }
}

/// Destination input widget
class DestinationInput extends StatelessWidget {
  final TextEditingController controller;
  final TextInputAction? textInputAction;
  final VoidCallback? onSubmitted;
  final bool enabled;

  const DestinationInput({
    super.key,
    required this.controller,
    this.textInputAction,
    this.onSubmitted,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return LocationInput(
      controller: controller,
      label: 'Destination',
      hint: 'e.g., Casablanca, Morocco',
      icon: Icons.location_on,
      iconColor: Colors.red,
      textInputAction: textInputAction,
      onSubmitted: onSubmitted,
      enabled: enabled,
    );
  }
}