import 'package:flutter/material.dart';

import '../models/traffic_prediction.dart';

/// Card displaying prediction results
class PredictionResultCard extends StatelessWidget {
  final TrafficPredictionResult prediction;

  const PredictionResultCard({
    super.key,
    required this.prediction,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(context),
            const Divider(height: 24),
            _buildSeverityBadge(context),
            const SizedBox(height: 16),
            if (prediction.traffic != null) _buildTrafficSection(context),
            if (prediction.weather != null) ...[
              const SizedBox(height: 16),
              _buildWeatherSection(context),
            ],
            if (prediction.events != null && prediction.events!.hasEvents) ...[
              const SizedBox(height: 16),
              _buildEventsSection(context),
            ],
            if (prediction.routing != null &&
                prediction.routing!.routes.isNotEmpty) ...[
              const SizedBox(height: 16),
              _buildRoutingSection(context),
            ],
            if (prediction.recommendations.isNotEmpty) ...[
              const SizedBox(height: 16),
              _buildRecommendations(context),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: _getSeverityColor().withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(
            _getSeverityIcon(),
            color: _getSeverityColor(),
            size: 28,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Traffic Analysis',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              if (prediction.summary.isNotEmpty)
                Text(
                  prediction.summary,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey[600],
                      ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSeverityBadge(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: _getSeverityColor().withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _getSeverityColor().withOpacity(0.5)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(_getSeverityIcon(), color: _getSeverityColor(), size: 16),
          const SizedBox(width: 8),
          Text(
            'Severity: ${prediction.overallSeverity.toUpperCase()}',
            style: TextStyle(
              color: _getSeverityColor(),
              fontWeight: FontWeight.bold,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrafficSection(BuildContext context) {
    final traffic = prediction.traffic!;

    return _buildSection(
      context,
      icon: '🚗',
      title: 'Traffic',
      children: [
        _buildInfoRow('Congestion', traffic.congestionLevel.toUpperCase()),
        _buildInfoRow('Delay', '${traffic.estimatedDelayMinutes} min'),
        _buildInfoRow('Confidence', traffic.confidence),
        if (traffic.recommendedAction.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  const Icon(Icons.lightbulb, color: Colors.blue, size: 16),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      traffic.recommendedAction,
                      style: const TextStyle(fontSize: 13),
                    ),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildWeatherSection(BuildContext context) {
    final weather = prediction.weather!;

    return _buildSection(
      context,
      icon: weather.weatherIcon,
      title: 'Weather',
      children: [
        _buildInfoRow('Condition', weather.description),
        _buildInfoRow('Temperature', '${weather.temperature.toStringAsFixed(1)}°C'),
        _buildInfoRow('Feels Like', '${weather.feelsLike.toStringAsFixed(1)}°C'),
        _buildInfoRow('Humidity', '${weather.humidity}%'),
        _buildInfoRow('Wind', '${weather.windSpeed.toStringAsFixed(1)} m/s'),
        if (weather.isHazardous)
          Container(
            margin: const EdgeInsets.only(top: 8),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.orange.shade50,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.orange.shade200),
            ),
            child: const Row(
              children: [
                Icon(Icons.warning_amber, color: Colors.orange, size: 20),
                SizedBox(width: 8),
                Text(
                  'Hazardous weather conditions',
                  style: TextStyle(color: Colors.orange, fontWeight: FontWeight.w500),
                ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _buildEventsSection(BuildContext context) {
    final events = prediction.events!;

    return _buildSection(
      context,
      icon: '📅',
      title: 'Events',
      children: [
        _buildInfoRow('Count', '${events.eventCount} events'),
        _buildInfoRow('Attendance', '~${events.totalExpectedAttendance}'),
        _buildInfoRow('Impact', events.level.toUpperCase()),
        if (events.message.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              events.message,
              style: const TextStyle(fontStyle: FontStyle.italic, fontSize: 13),
            ),
          ),
      ],
    );
  }

  Widget _buildRoutingSection(BuildContext context) {
    final routing = prediction.routing!;
    final recommended = routing.recommendedRoute;

    if (recommended == null) return const SizedBox.shrink();

    return _buildSection(
      context,
      icon: '🛣️',
      title: 'Recommended Route',
      children: [
        _buildInfoRow('Via', recommended.summary),
        _buildInfoRow('Distance', recommended.distanceText),
        _buildInfoRow('Duration', recommended.durationInTrafficText ?? recommended.durationText),
        if (recommended.delayMinutes > 0)
          _buildInfoRow('Traffic Delay', '+${recommended.delayMinutes} min'),
        if (routing.routes.length > 1)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              '${routing.routes.length - 1} alternative route(s) available',
              style: TextStyle(color: Colors.grey[600], fontSize: 12),
            ),
          ),
      ],
    );
  }

  Widget _buildRecommendations(BuildContext context) {
    return _buildSection(
      context,
      icon: '💡',
      title: 'Recommendations',
      children: prediction.recommendations
          .take(5)
          .map(
            (rec) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('• ', style: TextStyle(fontWeight: FontWeight.bold)),
                  Expanded(child: Text(rec, style: const TextStyle(fontSize: 13))),
                ],
              ),
            ),
          )
          .toList(),
    );
  }

  Widget _buildSection(
    BuildContext context, {
    required String icon,
    required String title,
    required List<Widget> children,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(icon, style: const TextStyle(fontSize: 18)),
            const SizedBox(width: 8),
            Text(
              title,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ...children,
      ],
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: TextStyle(color: Colors.grey[600], fontSize: 13),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontWeight: FontWeight.w500, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Color _getSeverityColor() {
    switch (prediction.severityLevel) {
      case SeverityLevel.high:
        return Colors.red;
      case SeverityLevel.medium:
        return Colors.orange;
      case SeverityLevel.low:
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  IconData _getSeverityIcon() {
    switch (prediction.severityLevel) {
      case SeverityLevel.high:
        return Icons.warning;
      case SeverityLevel.medium:
        return Icons.info;
      case SeverityLevel.low:
        return Icons.check_circle;
      default:
        return Icons.help_outline;
    }
  }
}