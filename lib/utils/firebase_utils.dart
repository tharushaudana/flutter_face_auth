import 'dart:developer';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:face_auth/pages/user_modal.dart';

Future<bool> downloadUsers(List<UserModal> list) async {
  final db = FirebaseFirestore.instance;

  list.clear();

  try {
    QuerySnapshot snapshot = await db.collection('users').get();

    for (var doc in snapshot.docs) {
      final user = UserModal();

      user.setName(doc.get('name'));
      user.setFaceData(doc.get('faceData'));

      list.add(user);
    }

    return true;
  } catch (e) {
    return false;
  }
}

Future<bool> pushNewUser(UserModal user) async {
  final db = FirebaseFirestore.instance;

  try {
    await db.collection('users').add({
      'name': user.name,
      'faceData': user.faceData,
    });

    return true;
  } catch (e) {
    log(e.toString());
    return false;
  }
}
