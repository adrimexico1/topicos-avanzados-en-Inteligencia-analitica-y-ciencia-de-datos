import 'package:flutter/material.dart';
import 'package:flutter_resnet_new/routes/Routes.dart';
import 'package:flutter_resnet_new/screens/ResnetScreen.dart';
import 'package:flutter_resnet_new/themes/Themes.dart';
import 'package:sizer/sizer.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {

  @override
  Widget build(BuildContext context) {

    return Sizer(builder: (context, orientation, device){
      return MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Resnet App',
        theme: CustomTheme().baseTheme,


        initialRoute: ResnetScreen.routeName,

        routes: routes,
      );
    });
  }
}
