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
    property var scaleVal: 1
    property int maxZ: 0

    //defaults
    /*state: "BASE"
    scale: scaleVal

    states: [
        State {
            name: "BASE"
            PropertyChanges {target: mouseArea; drag.target: undefined}
        },

        State {
            name: "DRAGGING"
            PropertyChanges {target: mouseArea; drag.target: jarvis}
        }
    ]*/

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
            if(jarvis.focus != true){
                jarvis.z = maxZ
                maxZ += 1
            }
            
            jarvis.focus = true

            /*if(parent.state == "BASE"){
                mouse.accepted = false
            }

            if(parent.state == "DRAGGING"){
                mouse.accepted = true
            }*/
        
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
    }

    //Defines key event changes
    Keys.onPressed: {
    
        //Control widget scaling with up and down keys  
        if(event.key == Qt.Key_Up){
            scaleVal = scaleVal + 0.05
        }


        if(event.key == Qt.Key_Down){
            scaleVal = scaleVal - 0.05
        }

        //Enable dragging when "Alt" is pressed
        /*if(event.key == Qt.Key_Alt){
            state = "DRAGGING"
        }*/

    }

    //Disable dragging when the "Alt" is not pressed
    /*Keys.onReleased:{
        if(event.key == Qt.Key_Alt){
            state = "BASE"
        }
    }*/
}