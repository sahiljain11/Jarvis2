import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.2
import "../components"

ApplicationWindow {
    title: qsTr("Calendar")
    width: 400
    height: 200
    visible: true

    FontLoader {
            id: techFont
            source: "PixelLCD.ttf"
        }

    Rectangle {
        anchors.fill: parent
        color: "white"
        opacity: 0.2
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            text: new Date().toLocaleTimeString(Qt.locale())
            font.family: techFont.name
        }
    }
}