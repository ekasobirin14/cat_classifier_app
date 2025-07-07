// ignore_for_file: depend_on_referenced_packages, use_build_context_synchronously

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const CatClassifierApp());
}

class CatClassifierApp extends StatelessWidget {
  const CatClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cat Classifier',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: const CatClassifierHomePage(),
    );
  }
}

class CatClassifierHomePage extends StatefulWidget {
  const CatClassifierHomePage({super.key});

  @override
  State<CatClassifierHomePage> createState() => _CatClassifierHomePageState();
}

class _CatClassifierHomePageState extends State<CatClassifierHomePage> {
  final ImagePicker _picker = ImagePicker();
  Interpreter? _interpreter;
  List<String> _labels = [];
  String _result = '';
  File? _imageFile;
  final int _inputSize = 224;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/model/modelmobilenetv2.tflite',
      );
      final labelData = await DefaultAssetBundle.of(
        context,
      ).loadString('assets/model/labels.txt');
      _labels = labelData.split('\n');
      debugPrint('Model and labels loaded');
    } catch (e) {
      debugPrint('Failed to load model: $e');
    }
  }

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final image = File(pickedFile.path);
      setState(() {
        _imageFile = image;
        _result = '';
      });
      _classifyImage(image);
    }
  }

  Future<void> _takePhoto() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      final image = File(pickedFile.path);
      setState(() {
        _imageFile = image;
        _result = '';
      });
      _classifyImage(image);
    }
  }

  Future<void> _classifyImage(File imageFile) async {
    if (_interpreter == null) {
      debugPrint('Interpreter not initialized');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    final imageBytes = imageFile.readAsBytesSync();
    final decodedImage = img.decodeImage(imageBytes);
    final resizedImage = img.copyResize(
      decodedImage!,
      width: _inputSize,
      height: _inputSize,
    );

    final input = Float32List(_inputSize * _inputSize * 3);
    int pixelIndex = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        input[pixelIndex++] = pixel.r / 255.0;
        input[pixelIndex++] = pixel.g / 255.0;
        input[pixelIndex++] = pixel.b / 255.0;
      }
    }

    var outputShape = _interpreter!.getOutputTensor(0).shape;
    var inputTensor = input.reshape([1, _inputSize, _inputSize, 3]);
    var outputTensor = List.filled(
      outputShape[1],
      0.0,
    ).reshape([1, outputShape[1]]);

    _interpreter!.run(inputTensor, outputTensor);

    int topLabelIndex = 0;
    double maxProb = 0.0;
    for (int i = 0; i < outputShape[1]; i++) {
      if (outputTensor[0][i] > maxProb) {
        maxProb = outputTensor[0][i];
        topLabelIndex = i;
      }
    }

    setState(() {
      _result =
          '${_labels[topLabelIndex]} (${(maxProb * 100).toStringAsFixed(2)}%)';
      _isLoading = false;
    });
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cat Breed Classifier'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_imageFile != null)
              Image.file(_imageFile!, height: 250)
            else
              Container(
                height: 250,
                color: Colors.grey[200],
                child: const Center(child: Text('No image selected')),
              ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _pickImage,
              icon: const Icon(Icons.image),
              label: const Text('Pick from Gallery'),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: _takePhoto,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take Photo'),
            ),
            const SizedBox(height: 16),
            _isLoading
                ? const CircularProgressIndicator()
                : Text(
                  _result.isEmpty ? 'Prediction will appear here' : _result,
                  style: const TextStyle(fontSize: 18),
                  textAlign: TextAlign.center,
                ),
          ],
        ),
      ),
    );
  }
}
