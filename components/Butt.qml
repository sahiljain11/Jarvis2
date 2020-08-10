import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12
import QtMultimedia 5.12
import QtQml.Models 2.12
import "../components"


//Define a Button widget that sends the signal 'touched' every time is pressed
Picture{
    id: back
    signal touched()
    signal inside()
    signal outside()
    property string word: ""
    property string color_: "black"
    property var picWidth: sourceWidth 
    property var picHeight: sourceHeight

    width: 200
    height: 200

    image: ""

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

        onClicked: {console.log("Clicked")}
        
    }

    /*Keys.onPressed:{
        console.log("pressed key")
        if (event.key == Qt.Key_Up){
            console.log("pressed 0")
            //mou.pressed()
        }
    }*/

    Component.onCompleted: {
        mou.clicked.connect(touched)
    }   
}