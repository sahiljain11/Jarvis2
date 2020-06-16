import QtQuick 2.5
import QtGraphicalEffects 1.12

Rectangle{
    property url image: "../images/play.png"
    property var sourceWidth: pic.sourceSize.width
    property var sourceHeight: pic.sourceSize.height
    property string tint: "transparent"
    

    width: sourceWidth
    height: sourceHeight

    color: "transparent"
    Image{
        id: pic
        source: image
        anchors.fill: parent
        mipmap: true
    }

    ColorOverlay {
        anchors.fill: pic
        source: pic
        color: tint 
    }
}