// ignore_for_file: depend_on_referenced_packages, use_build_context_synchronously, deprecated_member_use
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:shimmer/shimmer.dart';
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
      title: 'CAT CLASSIFIER',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFFFDF6EC),
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontFamily: 'Comic Sans', fontSize: 16),
          bodyMedium: TextStyle(fontFamily: 'Comic Sans', fontSize: 14),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(seconds: 2), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const CatClassifierHomePage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFDABFFF),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.pets, size: 80, color: Colors.deepPurple.shade400),
            const SizedBox(height: 24),
            Text(
              'Cat Breed Classifier',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.deepPurple.shade700,
                letterSpacing: 1.2,
              ),
            ),
          ],
        ),
      ),
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
  bool _isDarkMode = false;

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
      _labels = labelData.split('\n').map((e) => e.trim()).toList();
    } catch (e) {
      debugPrint('Failed to load model: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
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
    if (_interpreter == null) return;
    setState(() => _isLoading = true);

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

    var inputTensor = input.reshape([1, _inputSize, _inputSize, 3]);
    var outputShape = _interpreter!.getOutputTensor(0).shape;
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
          '${_labels[topLabelIndex]} 🐾 (${(maxProb * 100).toStringAsFixed(2)}%)';
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
    final catInfo = {
      'abyssinian': {
        'Asal': 'Ethiopia',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '9–15 tahun',
        'Persebaran': 'Seluruh dunia',
        'FunFact':
            'Salah satu ras kucing tertua, sangat aktif dan suka memanjat!',
      },
      'bengal': {
        'Asal': 'Amerika Serikat',
        'Ukuran Rata-rata': 'Sedang hingga besar',
        'Umur Rata-rata': '12–16 tahun',
        'Persebaran': 'Seluruh dunia',
        'FunFact': 'Punya pola bulu eksotis seperti macan tutul mini.',
      },
      'birman': {
        'Asal': 'Myanmar (Burma)',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '12–16 tahun',
        'Persebaran': 'Global',
        'FunFact': 'Dikenal dengan "sarung tangan putih" di kaki mereka.',
      },
      'bombay': {
        'Asal': 'Amerika Serikat',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '12–15 tahun',
        'Persebaran': 'Terutama di Amerika',
        'FunFact': 'Bulu hitam pekatnya bikin dia mirip mini panther.',
      },
      'british shorthair': {
        'Asal': 'Inggris',
        'Ukuran Rata-rata': 'Sedang hingga besar',
        'Umur Rata-rata': '14–20 tahun',
        'Persebaran': 'Eropa dan global',
        'FunFact': 'Pipi gembul dan ekspresi cueknya bikin gampang disayang!',
      },
      'egyptian mau': {
        'Asal': 'Mesir',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '12–15 tahun',
        'Persebaran': 'Terbatas tapi berkembang',
        'FunFact': 'Satu-satunya kucing dengan bintik alami sejak lahir.',
      },
      'maine coon': {
        'Asal': 'Amerika Serikat',
        'Ukuran Rata-rata': 'Besar',
        'Umur Rata-rata': '12–15 tahun',
        'Persebaran': 'Global',
        'FunFact': 'Termasuk kucing terbesar, suka air, dan sangat ramah.',
      },
      'persian': {
        'Asal': 'Iran (Persia)',
        'Ukuran Rata-rata': 'Sedang hingga besar',
        'Umur Rata-rata': '12–17 tahun',
        'Persebaran': 'Seluruh dunia',
        'FunFact': 'Dikenal karena bulu panjang dan wajah datar khasnya.',
      },
      'ragdoll': {
        'Asal': 'Amerika Serikat',
        'Ukuran Rata-rata': 'Besar',
        'Umur Rata-rata': '13–18 tahun',
        'Persebaran': 'Global',
        'FunFact': 'Saat digendong, tubuhnya jadi “lemas” kayak boneka!',
      },
      'russian blue': {
        'Asal': 'Rusia',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '15–20 tahun',
        'Persebaran': 'Eropa dan Amerika',
        'FunFact': 'Punya bulu keperakan dan sangat setia pada satu pemilik.',
      },
      'siamese': {
        'Asal': 'Thailand',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '15–20 tahun',
        'Persebaran': 'Global',
        'FunFact': 'Sangat vokal dan suka “ngobrol” dengan pemiliknya!',
      },
      'sphynx': {
        'Asal': 'Kanada',
        'Ukuran Rata-rata': 'Sedang',
        'Umur Rata-rata': '8–14 tahun',
        'Persebaran': 'Global (dengan perawatan khusus)',
        'FunFact': 'Gak punya bulu! Tapi hangat dan suka pelukan.',
      },
    };

    String breedName =
        _result.isNotEmpty
            ? _result.split(' 🐾').first.trim().toLowerCase()
            : '';
    final info = catInfo[breedName];

    // Mapping label ke icon
    final infoIcons = {
      'Asal': Icons.flag,
      'Ukuran Rata-rata': Icons.straighten,
      'Umur Rata-rata': Icons.cake,
      'Persebaran': Icons.public,
      'FunFact': Icons.lightbulb,
    };

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.exit_to_app),
          tooltip: 'Exit',
          onPressed: () {
            exit(0); // import 'dart:io'; sudah ada di atas
          },
        ),
        backgroundColor:
            _isDarkMode ? Colors.grey[900] : const Color(0xFFDABFFF),
        foregroundColor: _isDarkMode ? Colors.white : Colors.black,
        title: const Text(
          '🐱 Cat Breed Classifier',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(_isDarkMode ? Icons.light_mode : Icons.dark_mode),
            onPressed: () {
              setState(() {
                _isDarkMode = !_isDarkMode;
              });
            },
          ),
        ],
        elevation: 4,
        shadowColor: Colors.deepPurple.withOpacity(0.2),
      ),
      backgroundColor: _isDarkMode ? Colors.black : const Color(0xFFFDF6EC),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final isWide = constraints.maxWidth > 600;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Center(
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  maxWidth: isWide ? 500 : double.infinity,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 400),
                      curve: Curves.easeInOut,
                      height: isWide ? 320 : 250,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: Colors.deepPurple.shade100,
                          width: 2,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.deepPurple.withOpacity(0.07),
                            blurRadius: 16,
                            offset: const Offset(0, 8),
                          ),
                        ],
                      ),
                      child:
                          _imageFile != null
                              ? ClipRRect(
                                borderRadius: BorderRadius.circular(18),
                                child: Stack(
                                  children: [
                                    Shimmer.fromColors(
                                      baseColor: Colors.deepPurple.shade50,
                                      highlightColor:
                                          Colors.deepPurple.shade100,
                                      child: Container(
                                        width: double.infinity,
                                        height: double.infinity,
                                        color: Colors.white,
                                      ),
                                    ),
                                    Image.file(
                                      _imageFile!,
                                      fit: BoxFit.cover,
                                      width: double.infinity,
                                      height: double.infinity,
                                    ),
                                  ],
                                ),
                              )
                              : Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(
                                      Icons.image_outlined,
                                      size: 60,
                                      color: Colors.deepPurple.shade100,
                                    ),
                                    const SizedBox(height: 8),
                                    Text(
                                      'No image selected 🐾',
                                      style: TextStyle(
                                        color: Colors.deepPurple.shade200,
                                        fontSize: 16,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                    ),
                    const SizedBox(height: 28),
                    Wrap(
                      spacing: 16,
                      runSpacing: 10,
                      alignment: WrapAlignment.center,
                      children: [
                        ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.deepPurple,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 14,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 3,
                          ),
                          onPressed: () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.image),
                          label: const Text('From Gallery'),
                        ),
                        ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange.shade400,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 14,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 3,
                          ),
                          onPressed: () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Take Photo'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 28),
                    _isLoading
                        ? const CircularProgressIndicator()
                        : Column(
                          children: [
                            AnimatedDefaultTextStyle(
                              duration: const Duration(milliseconds: 300),
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color:
                                    _result.isEmpty
                                        ? Colors.deepPurple.shade200
                                        : Colors.deepPurple.shade700,
                              ),
                              child: Text(
                                _result.isEmpty
                                    ? 'Let\'s see what breed this cat is 🐱'
                                    : _result,
                                textAlign: TextAlign.center,
                              ),
                            ),
                            const SizedBox(height: 18),
                            if (info != null)
                              Card(
                                elevation: 6,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                color:
                                    _isDarkMode
                                        ? Colors.grey[900]
                                        : Colors.white,
                                shadowColor: Colors.deepPurple.withOpacity(
                                  0.08,
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.all(20),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children:
                                        info.entries.map((entry) {
                                          final icon =
                                              infoIcons[entry.key] ??
                                              Icons.info_outline;
                                          return Padding(
                                            padding: const EdgeInsets.only(
                                              bottom: 12.0,
                                            ),
                                            child: Row(
                                              crossAxisAlignment:
                                                  CrossAxisAlignment.start,
                                              children: [
                                                Container(
                                                  decoration: BoxDecoration(
                                                    color:
                                                        Colors
                                                            .deepPurple
                                                            .shade50,
                                                    borderRadius:
                                                        BorderRadius.circular(
                                                          8,
                                                        ),
                                                  ),
                                                  padding: const EdgeInsets.all(
                                                    6,
                                                  ),
                                                  child: Icon(
                                                    icon,
                                                    color:
                                                        Colors
                                                            .deepPurple
                                                            .shade400,
                                                    size: 22,
                                                  ),
                                                ),
                                                const SizedBox(width: 12),
                                                Expanded(
                                                  child: RichText(
                                                    text: TextSpan(
                                                      style: TextStyle(
                                                        fontSize: 15,
                                                        fontFamily:
                                                            'Comic Sans',
                                                        color:
                                                            _isDarkMode
                                                                ? Colors.white
                                                                : Colors
                                                                    .black87,
                                                      ),
                                                      children: [
                                                        TextSpan(
                                                          text:
                                                              '${entry.key}: ',
                                                          style: TextStyle(
                                                            fontWeight:
                                                                FontWeight.bold,
                                                            color:
                                                                Colors
                                                                    .deepPurple
                                                                    .shade400,
                                                          ),
                                                        ),
                                                        TextSpan(
                                                          text: entry.value,
                                                          style: TextStyle(
                                                            color:
                                                                Colors
                                                                    .orange
                                                                    .shade700,
                                                            fontWeight:
                                                                entry.key ==
                                                                        'FunFact'
                                                                    ? FontWeight
                                                                        .w600
                                                                    : FontWeight
                                                                        .normal,
                                                          ),
                                                        ),
                                                      ],
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                          );
                                        }).toList(),
                                  ),
                                ),
                              ),
                          ],
                        ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
