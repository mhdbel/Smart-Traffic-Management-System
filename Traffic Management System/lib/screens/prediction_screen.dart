import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/traffic_provider.dart';
import '../widgets/prediction_result_card.dart';
import '../widgets/loading_indicator.dart';

/// Screen showing detailed prediction results
class PredictionScreen extends StatelessWidget {
  final String origin;
  final String destination;

  const PredictionScreen({
    super.key,
    required this.origin,
    required this.destination,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Prediction Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => context.read<TrafficProvider>().retry(),
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: Consumer<TrafficProvider>(
        builder: (context, provider, _) {
          switch (provider.state) {
            case LoadingState.loading:
              return const LoadingIndicator(
                message: 'Analyzing traffic conditions...',
              );
            case LoadingState.error:
              return _buildError(context, provider);
            case LoadingState.loaded:
              return _buildResults(context, provider);
            default:
              return const Center(
                child: Text('No data available'),
              );
          }
        },
      ),
    );
  }

  Widget _buildError(BuildContext context, TrafficProvider provider) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 64,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(height: 16),
            Text(
              'Error',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              provider.error ?? 'An unknown error occurred',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: () => provider.retry(),
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Go Back'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResults(BuildContext context, TrafficProvider provider) {
    final prediction = provider.prediction!;

    return RefreshIndicator(
      onRefresh: () => provider.retry(),
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Route info card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.route, size: 20),
                        const SizedBox(width: 8),
                        Text(
                          'Route',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    _buildRouteStep(
                      context,
                      Icons.trip_origin,
                      'From',
                      origin,
                      Colors.green,
                    ),
                    Container(
                      margin: const EdgeInsets.only(left: 11),
                      height: 20,
                      width: 2,
                      color: Colors.grey.shade300,
                    ),
                    _buildRouteStep(
                      context,
                      Icons.location_on,
                      'To',
                      destination,
                      Colors.red,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Main prediction result
            PredictionResultCard(prediction: prediction),

            const SizedBox(height: 16),

            // Execution info
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Analysis completed in ${prediction.executionTimeMs.toStringAsFixed(0)}ms',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.grey,
                          ),
                    ),
                    Text(
                      '${prediction.successfulAgents}/4 agents',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.grey,
                          ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRouteStep(
    BuildContext context,
    IconData icon,
    String label,
    String location,
    Color color,
  ) {
    return Row(
      children: [
        Icon(icon, color: color, size: 24),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.grey,
                    ),
              ),
              Text(
                location,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        ),
      ],
    );
  }
}