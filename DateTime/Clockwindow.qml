import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.2
import "../components"




JarvisWidget {
    width: 400
    height: 200

    FontLoader {
            id: techFont
            source: "PixelLCD.ttf"
        }
    Rectangle {
        anchors.fill: parent
        opacity: 0
        Text {
            id: timee
            color: "white"
            x: 10
            y: 10
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            font.pointSize: 24
            font.family: techFont.name
            text: Qt.formatTime(new Date(),"hh:mm:ss")
        }
    }


    Timer {
        id: timer
        interval: 1000
        repeat: true
        running: true

        onTriggered:
        {
            timee.text =  Qt.formatTime(new Date(),"hh:mm:ss")
        }
    }
}