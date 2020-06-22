import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12

Picture{
    id: back
    signal touched()
    signal inside()
    signal outside()
    property var picWidth: sourceWidth/2
    property var picHeight: sourceHeight/2
    property bool vis: false


    image: "../images/email_preview_button.png"

    Text{
        id: title
        text: 'hello world'
        anchors.left: parent.left
        anchors.leftMargin:parent.width/15
        anchors.top : parent.top
        anchors.topMargin : parent.height/15
        font.pointSize: parent.height/22
    }
    Text{
        id: sender
        text: 'conradiste@gmail.com'
        anchors{
            top: title.bottom
            topMargin: parent.height/20
            left: title.left
        }
        font.pointSize: parent.height/25
    }
    Text{
        id: preview_email
        clip : true
        anchors{
               left: title.left
               right: parent.right
               rightMargin: parent.width/20
               top: sender.bottom
               topMargin: parent.height/20
               bottom: parent.bottom
               bottomMargin: parent.height/7
        }
        font.pointSize: parent.height/25
        width: parent.width
        height: parent.height/2
        elide: Text.elideRight
         wrapMode: Text.WordWrap
         text: "preview email new line af 33 charf jsdl kfja lsd jfwio  lwadfs dfajio wjd foawnefo nawd lfnhqo nwglnWIO PNWL HOIF OWIF HOWAGN Olj wodjfowan flwn gownag"
    }

    Rectangle{
        id: mask
        color:'black'
        opacity: .3
        anchors.fill: parent
        visible: vis
        radius:19.78
    }
    //Darkens the button when the mouse is over it
    //Connects the clicked signal to 'touched'
    MouseArea{
        id: mou
        anchors.fill: parent
        hoverEnabled: true

        onEntered: {vis = true; entered.connect(inside);}

        onExited: {vis = false; exited.connect(outside)}

        onClicked: clicked.connect(touched)
    }
}