import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12
import QtMultimedia 5.0
import "./components"

// Initialize application window
ApplicationWindow{
    visible: true
    width: 600
    height: 600
    title: qsTr("Gmail")

    //background
    Image{
        id: grad
        source: "images/grad.png"
        anchors.fill: parent
    }

    // Container for Spotify widget
    Item{
        id: spotify_widget

        //Load fonts
        FontLoader { id: nidsans; source: "./fonts/Nidsans.ttf"}
        FontLoader{ id: astro; source: "./fonts/AstroSpace.ttf"}

        //custom properties
        property var scaleVal: 1
        //positioning
        width: 1200/1.5
        height: 550/1.5

        //defaults
        state: "BASE"
        scale: scaleVal

        states: [
            State {
                name: "BASE"
                PropertyChanges {target: mouseArea; drag.target: undefined}
            },

            State {
                name: "DRAGGING"
                PropertyChanges {target: mouseArea; drag.target: spotify_widget}
            }
        ]

        // Sets the background image
        Picture{
            id: back
            image: "images/frame1.png"
            anchors.fill: parent
            smooth: true
            opacity: 1
            focus: true
            //tint: "white"

            Rectangle{
                z: -2
                anchors.fill: parent
                color: "#00FFF5"
                opacity: 0.2
            }
        }

        //container for email buttons
        ScrollView{
            id: email_scroll
            Item{
                id: email_container
                anchors{
                    left: parent.left
                    right: parent.right
                    top: parent.top
                    bottom: parent.bottom
                    }
                    Gmail_Butt{
                        anchors{
                            right: parent.left


                        }
                        width: parent.width/3
                        height: parent.height/3

                        }
                    Gmail_Butt{

                        anchors{
                            right: parent.left
                            rightMargin: 20
                            bottom: parent.bottom
                            bottomMargin: parent.height/2
                        }

                        width: parent.width/3
                        height: parent.height/3

                        }

                }
            verticalScrollBarPolicy: Qt.ScrollBarAlwaysOn
            frameVisible: True
            anchors{
                right: parent.right
                rightMargin: parent.width/2
                top: parent.top
                bottom: parent.bottom
                left: parent.left
            }

            height: parent.height
            width: parent.width/2

        }


        //Text input for gmail
        TextField{
            id: textInput
            text: "find email"
            anchors{
                top: parent.top
                topMargin: parent.height/8
                right: parent.right
                rightMargin: parent.width/10
                left: parent.left
                leftMargin: parent.width/2
            }
            font.family: astro.name
            width: parent.width/10
            height: parent.height/10
        }

        //Sets up dragging functionality of window
        MouseArea{
            id: mouseArea
            anchors.fill: parent
            propagateComposedEvents: true

            onClicked: {
                mouse.accepted = false
            }

            //Allow mouse pressed events through if we are in the base state
            onPressed: {
                if(parent.state == "BASE"){
                    mouse.accepted = false
                }
                else{
                    mouse.accepted = true
                }

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

            if(event.key == Qt.Key_Left){
                console.log("Prese")

            }

            //Enable dragging when "Shift" is pressed
            if(event.key == Qt.Key_Shift){
                state = "DRAGGING"
            }
        }

        //Disable dragging when the "Shift" is not pressed
        Keys.onReleased:{
            if(event.key == Qt.Key_Shift){
                state = "BASE"
            }
        }
    }
}