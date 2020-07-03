import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.11


//Define a Button widget that sends the signal 'touched' every time is pressed
Picture{
    id: back
    signal touched()
    signal inside()
    signal outside()
    property string word: ""
    property var picWidth: sourceWidth 
    property var picHeight: sourceHeight

    image: "../images/play.png"

    //Places text if any inside the button
    Text {
        text: word 
        anchors.centerIn: parent
    }

    //Darkens the button when the mouse is over it
    //Connects the clicked signal to 'touched'
    MouseArea{
        id: mou
        anchors.fill: parent
        hoverEnabled: true

        onEntered: {back.tint = "#80800000"; entered.connect(inside);}

        onExited: {back.tint = "transparent"; exited.connect(outside)}
        
    }

    Component.onCompleted: {
        mou.clicked.connect(touched)
    }   
}