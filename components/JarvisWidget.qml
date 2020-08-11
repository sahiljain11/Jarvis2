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
    scale: scaleVal
    state: "closed"

    //Test box
    /*Rectangle{
        id: testBox
        property alias word: textBox.text
        x: 0
        y: 0
        width: 50
        height: 50
        color: "red"

        Text{
            id: textBox
            text: ""
        }
    }*/

    states: [ 
        State{
            name: "opened"
            PropertyChanges { target: jarvis; scale: 1}
        },
        State {
            name: "closed"
            PropertyChanges { target: jarvis; x: parent.width/2 - width/2; y: parent.height/2.6 - height/2; scale: 0}
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
            console.log("This widget was clicked")
            jarvis.focus = true 
            mouse.accepted = false
        }

        //Allow mouse pressed events through if we are in the base state
        onPressed: {

            // Debugging 
            console.log("Widget Pressed")
            //testBox.x = mouse.x
            //testBox.y = mouse.y
            //testBox.word = mouse.x.toString()  + " " + mouse.y.toString()
            

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
            console.log("A new widget was entered")
            jarvis.focus = true
            mouseArea.entered.connect(gainedFocus)
        }

        onReleased:{
            //console.log("This widget was released at (", mouse.x, ",", mouse.y, ")")
            drag.target = undefined
            mouse.accepted = false
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
    }
}