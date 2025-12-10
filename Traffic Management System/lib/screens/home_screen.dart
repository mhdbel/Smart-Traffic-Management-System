// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/traffic_provider.dart';
import '../widgets/prediction_result_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _formKey = GlobalKey<FormState>();
  final _originController = TextEditingController();
  final _destinationController = TextEditingController();

  @override
  void dispose() {
    _originController.dispose();
    _destinationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Smart Traffic'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showAboutDialog,
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildInputForm(),
              const SizedBox(height: 24),
              _buildResultSection(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInputForm() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Enter Route Details',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _originController,
                decoration: const InputDecoration(
                  labelText: 'Origin',
                  hintText: 'e.g., Rabat, Morocco',
                  prefixIcon: Icon(Icons.trip_origin),
                ),
                textInputAction: TextInputAction.next,
                validator: _validateLocation,
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _destinationController,
                decoration: const InputDecoration(
                  labelText: 'Destination',
                  hintText: 'e.g., Casablanca, Morocco',
                  prefixIcon: Icon(Icons.location_on),
                ),
                textInputAction: TextInputAction.done,
                onFieldSubmitted: (_) => _handleSubmit(),
                validator: _validateLocation,
              ),
              const SizedBox(height: 24),
              Consumer<TrafficProvider>(
                builder: (context, provider, _) {
                  final isLoading = provider.state == LoadingState.loading;
                  
                  return FilledButton.icon(
                    onPressed: isLoading ? null : _handleSubmit,
                    icon: isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.search),
                    label: Text(isLoading ? 'Analyzing...' : 'Get Prediction'),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultSection() {
    return Consumer<TrafficProvider>(
      builder: (context, provider, _) {
        switch (provider.state) {
          case LoadingState.initial:
            return _buildInitialState();
          case LoadingState.loading:
            return _buildLoadingState();
          case LoadingState.loaded:
            return PredictionResultCard(prediction: provider.prediction!);
          case LoadingState.error:
            return _buildErrorState(provider.error!);
        }
      },
    );
  }

  Widget _buildInitialState() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          children: [
            Icon(
              Icons.directions_car,
              size: 64,
              color: Theme.of(context).colorScheme.primary.withOpacity(0.5),
            ),
            const SizedBox(height: 16),
            Text(
              'Enter your route to get traffic predictions',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: Colors.grey,
                  ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingState() {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(32),
        child: Column(
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Analyzing traffic conditions...'),
            SizedBox(height: 8),
            Text(
              'Checking weather, events, and routes',
              style: TextStyle(color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorState(String error) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Icon(
              Icons.error_outline,
              size: 48,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(height: 16),
            Text(
              error,
              style: TextStyle(
                color: Theme.of(context).colorScheme.onErrorContainer,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: _handleSubmit,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }

  String? _validateLocation(String? value) {
    if (value == null || value.trim().isEmpty) {
      return 'This field is required';
    }
    if (value.trim().length < 2) {
      return 'Must be at least 2 characters';
    }
    if (value.trim().length > 200) {
      return 'Too long (max 200 characters)';
    }
    return null;
  }

  void _handleSubmit() {
    if (_formKey.currentState!.validate()) {
      context.read<TrafficProvider>().fetchPrediction(
            _originController.text.trim(),
            _destinationController.text.trim(),
          );
    }
  }

  void _showAboutDialog() {
    showAboutDialog(
      context: context,
      applicationName: 'Smart Traffic',
      applicationVersion: '1.0.0',
      applicationLegalese: '© 2024 Traffic Management System',
      children: [
        const SizedBox(height: 16),
        const Text(
          'An intelligent traffic management system that analyzes '
          'traffic, weather, and events to provide smart recommendations.',
        ),
      ],
    );
  }
}