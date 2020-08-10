import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12

    ScrollView {
        id: flick
        property var send: "" 
        property var msg: ""
        width: 1000; height: 5000;
        //contentWidth: sender.paintedWidth
        //contentHeight: sender.paintedHeight
        clip: true
        ScrollBar.vertical.interactive: true
        //Background
        Rectangle{
            id: back
            color:"#00FFF5"
            width: flick.width
            height: flick.height
            opacity: 0.4

            Component.onCompleted:{
                width = (sender.paintedWidth > flick.width) ? sender.paintedWidth : flick.width
                height = (sender.paintedHeight > flick.height) ? sender.paintedHeight : flick.height
            }
        }

        Rectangle {
            id: bord
            color: "transparent"
            border.color: "orange"
            border.width: 1
            anchors.fill: back
        }

        TextArea{
            id: sender
            readOnly: true
            //anchors.top: parent.top
            //anchors.topMargin: parent.height/10
            focus: true
            wrapMode: TextEdit.WrapAnywhere
            text: msg
        }
    }
