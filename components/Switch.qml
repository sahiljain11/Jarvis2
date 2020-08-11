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
    signal inside()
    property string word: ""
    property var picWidth: sourceWidth 
    property var picHeight: sourceHeight
    property bool isOn: false
    property bool doTint: true
    property url iconOn: ""
    property url iconOff: ""
    property alias rad: mask.radius

    state: "OFF"

    //Places text if any inside the switch
    Text {
        text: word
        anchors.fill: parent
        anchors.centerIn: parent
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        color: "white"
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

    Rectangle{
        id: mask
        visible: false
        anchors.fill: parent
        color: "#202044"
        opacity: 0.5
        radius: rad
    }

    //Defines mouse interatcions with the switch
    MouseArea{
        id: mou
        anchors.fill: parent
        hoverEnabled: true

        // Darken the icon when moving over it
        onEntered: {
            //the flag for tinting is set
            if(doTint){
                back.tint = "#80800000"
            }
            //the flag for a black rectangle is set
            else{
                mask.visible = true
            }
        }

        onExited: {
            if(doTint){
                back.tint = "transparent"
            }
            else{
                mask.visible = false
            }
        }

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