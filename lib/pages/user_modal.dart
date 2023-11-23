class UserModal {
  String? name;
  List? faceData;

  void setName(String n) {
    name = n;
  }

  void setFaceData(List data) {
    faceData = List.from(data);
  }
}