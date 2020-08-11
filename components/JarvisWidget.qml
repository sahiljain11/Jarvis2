import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12
import QtMultimedia 5.12
import QtQml.Models 2.12
import "../components"

Item{
    id: jarvis
    width: 600
    height: 600
    signal gainedFocus()
    //custom properties
    scale: 0
    state: "closed"

    states: [ 
        State{
            name: "opened"
            PropertyChanges { target: jarvis; scale: 1; z: parent.maxZ}
        },
        State {
            name: "closed"
            PropertyChanges { target: jarvis; x: parent.width/2 - width/2; y: parent.height/2.6 - height/2; scale: 0; z: -5}
        }
    ]

    transitions: [ 
        Transition {
            from: "closed"
            to: "opened"
            reversible: true
            PropertyAnimation {
                target: jarvis
                property: "scale"
                easing.bezierCurve: [0.175, 0.885, 0.32, 1.27, 1, 1]
                duration: 700
            }
        }
    ]

    //Sets up dragging functionality of window
    MouseArea{
        id: mouseArea
        anchors.fill: parent
        propagateComposedEvents: true
        hoverEnabled: true
        
        onClicked: {
            jarvis.focus = true 
            mouse.accepted = false
        }

        //Allow mouse pressed events through if we are in the base state
        onPressed: {
            
            if (mouse.modifiers == Qt.AltModifier){
                drag.target = jarvis
                mouse.accepted = true
            }

            else{
                drag.target = undefined
                mouse.accepted = false
            }
        }

        onEntered:{
            jarvis.focus = true
            mouseArea.entered.connect(gainedFocus)
        }

        onReleased:{
            drag.target = undefined
            mouse.accepted = false
        }
    }

    //Defines key event changes
    Keys.onPressed: {
        
        //Do not allow jarvis to be scaled lower than zero
        if (jarvis.scale <= 0){
            jarvis.scale = 0
            jarvis.state = "closed"
        }

        //Control widget scaling with up and down keys  
        else if(event.key == Qt.Key_Up){
            scale = scale + 0.05
        }

        else if(event.key == Qt.Key_Down){
            scale = scale - 0.05
        }
    }
}