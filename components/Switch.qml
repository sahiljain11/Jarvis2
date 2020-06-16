import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.11


// Defines a switch
// You must input an image for when the switch is "ON" and one for when it is "OFF"
// It is by default "OFF"
// The state of the swithc can be accessed with "state"
Picture{
    id: back
    signal touched()
    property string word: ""
    property var picWidth: sourceWidth 
    property var picHeight: sourceHeight
    property bool isOn: false
    property url iconOn: ""
    property url iconOff: ""

    state: "OFF"

    //Places text if any inside the switch
    Text {
        text: word
        anchors.centerIn: parent 
    }

    //Define the on/off states of the switch
    states: [
        State{
            name: "ON"
            PropertyChanges {target: back; image: iconOn}  
        },

        State{
            name: "OFF"
            PropertyChanges {target: back; image: iconOff}
        }
    ]

    //Defines mouse interatcions with the switch
    MouseArea{
        id: mou
        anchors.fill: parent
        hoverEnabled: true

        // Darken the icon when moving over it
        onEntered: back.tint = "#80800000"
        onExited: back.tint = "transparent"

        onClicked: {

            //Change the on/off status of the switch
            if (back.state == "ON"){
                back.state =  "OFF"
            }
            
            else{
                back.state = "ON"
            }

            //Connect the clicked signal with touched
            clicked.connect(touched)
        }
    }   
}