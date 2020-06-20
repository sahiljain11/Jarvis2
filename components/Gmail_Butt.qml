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
        anchors.topMargin : 30
    }
    Text{
        id: sender
        text: 'conradiste@gmail.com'
        anchors{
            top: title.bottom
            topMargin: 15
            left: title.left
        }
    }
    Text{
        id: preview_email

        anchors{
               left: title.left
               right: parent.right
               rightMargin: 30
               top: sender.bottom
               topMargin: 40

        }
         wrapMode: Text.WordWrap
         text: "this is a test preview email. this should not be the full email"
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