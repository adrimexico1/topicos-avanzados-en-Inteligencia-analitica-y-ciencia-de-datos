import 'dart:convert';
import 'dart:io';
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:sizer/sizer.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:rflutter_alert/rflutter_alert.dart';

class ResnetScreen extends StatefulWidget {
  static String routeName = 'ResnetScreen';

  @override
  _ResnetScreenState createState() => _ResnetScreenState();
}

class _ResnetScreenState extends State<ResnetScreen> {
  File? _image;
  String? _pathImg;

  final url = Uri.parse("https://tensorflow-serving-persons-latest.onrender.com/v1/models/persons:predict"); 
  // final url = Uri.parse("https://flowers-serving-l4tu.onrender.com/v1/models/flowers_model:predict"); //
  final headers = {"Content-Type": "application/json;charset=UTF-8"};

  @override
  void initState() {
    super.initState();
  }

  Future<void> getImage(ImageSource source) async {
    try {
      final image = await ImagePicker().pickImage(source: source);
      if (image == null) return;

      final imagePermanent = await saveFilePermanently(image.path);
      _pathImg = image.path;

      setState(() {
        _image = imagePermanent;
      });
    } on PlatformException catch (e) {
      print("Falló al obtener imagen: $e");
    }
  }

  Future<File> saveFilePermanently(String imagePath) async {
    final directory = await getApplicationDocumentsDirectory();
    final name = path.basename(imagePath);
    final image = File('${directory.path}/$name');
    return File(imagePath).copy(image.path);
  }

  // 🔥🔥🔥 CORREGIDO: convertir imagen a GRAYSCALE 150×150×1 normalizada
  Future<List<List<List<double>>>> processImage(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? image = img.decodeImage(bytes);

    if (image == null) return [];

    // Convertir a escala de grises
    img.Image grayscale = img.grayscale(image);

    // Redimensionar a 150x150 según tu modelo
    img.Image resized = img.copyResize(grayscale, width: 150, height: 150);

    // Convertir a 150×150×1 normalizado (0–1)
    List<List<List<double>>> result = List.generate(
      150,
      (i) => List.generate(
        150,
        (j) {
          int pixel = resized.getPixel(j, i);
          double gray = img.getLuminance(pixel) / 255.0;
          return [gray]; // canal único
        },
      ),
    );

    return result;
  }

  Future<void> uploadImage() async {
    if (_pathImg == null) return;

    // Mostrar loading
    showDialog(
      context: context,
      builder: (context) => const Center(child: CircularProgressIndicator()),
    );

    try {
      final processedImage = await processImage(File(_pathImg!));

      // Crear JSON para TensorFlow Serving
      final predictionInstance = {
        "instances": [processedImage]
      };

      final res = await http.post(
        url,
        headers: headers,
        body: jsonEncode(predictionInstance),
      );

      Navigator.pop(context);

      if (res.statusCode == 200) {
        final jsonPrediction = jsonDecode(res.body);
        final pred = jsonPrediction['predictions'][0] as List;

        // Obtener el índice con mayor probabilidad
        final maxIndex = pred.indexOf(pred.reduce((a, b) => a > b ? a : b));

        // Leer labels desde index.json
        final value = await rootBundle.loadString('assets/json/index.json');
        var data = json.decode(value);
        var classResultPrediction = data[maxIndex.toString()][1];

        Alert(
          context: context,
          type: AlertType.success,
          title: "¡Predicción exitosa!",
          desc: "Clase: $classResultPrediction\n(Proba: ${pred[maxIndex].toStringAsFixed(4)})",
          buttons: [
            DialogButton(
              child: const Text(
                "Aceptar",
                style: TextStyle(color: Colors.white, fontSize: 20),
              ),
              onPressed: () => Navigator.pop(context),
              color: const Color(0xFF02B1FF),
              width: 120,
            )
          ],
        ).show();
      } else {
        Alert(
          context: context,
          type: AlertType.error,
          title: "Error",
          desc: "Ocurrió un error al mandar la imagen",
          buttons: [
            DialogButton(
              child: const Text(
                "Aceptar",
                style: TextStyle(color: Colors.white, fontSize: 20),
              ),
              onPressed: () => Navigator.pop(context),
              color: const Color(0xFF02B1FF),
              width: 120,
            )
          ],
        ).show();
      }
    } catch (e) {
      Navigator.pop(context);
      print(e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    const kOtherColor = Color(0xFFEFEFEF);
    const kTopBorderRadius = BorderRadius.only(
      topLeft: Radius.circular(20),
      topRight: Radius.circular(20),
    );
    const sizedBox = SizedBox(height: 4);

    return GestureDetector(
      onTap: () => FocusManager.instance.primaryFocus?.unfocus(),
      child: Scaffold(
        floatingActionButton: Visibility(
          visible: _pathImg != null,
          child: FloatingActionButton(
            child: const Icon(Icons.send_outlined, color: Colors.white),
            backgroundColor: const Color(0xFF02B1FF),
            onPressed: uploadImage,
          ),
        ),
        body: Column(
          children: [
            Container(
              width: 100.w,
              height: 10.h,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'ADRIMEXICO1 Reconocimiento de Personas',
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                      sizedBox,
                    ],
                  ),
                ],
              ),
            ),
            Expanded(
              child: Container(
                padding: EdgeInsets.symmetric(horizontal: 15.w),
                decoration: BoxDecoration(
                  color: kOtherColor,
                  borderRadius: kTopBorderRadius,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 50),
                    _image != null
                        ? Image.file(
                            _image!,
                            width: 300,
                            height: 400,
                            fit: BoxFit.cover,
                          )
                        : Image.asset('assets/images/take_photo.gif'),
                    const SizedBox(height: 20),
                    ElevatedButton.icon(
                      onPressed: () => getImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: const Text('Cámara'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF02B1FF),
                        padding: const EdgeInsets.all(20),
                        textStyle: const TextStyle(fontSize: 14, color: Colors.white),
                      ),
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton.icon(
                      onPressed: () => getImage(ImageSource.gallery),
                      icon: const Icon(Icons.attach_file_sharp),
                      label: const Text('Galería'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF02B1FF),
                        padding: const EdgeInsets.all(20),
                        textStyle: const TextStyle(fontSize: 14, color: Colors.white),
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
}