import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtCharts 2.0
import QtPositioning 5.12
import QtLocation 5.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.12
import QtQml.Models 2.12
import "./components"
import "./Spotify"
import "./Gmail"
import "./Weather"
import "./Calendar"
import "./CovidGraphing"
import "./DateTime"

Item{
    id: window
    visible: true
    width: 1920
    height: 1080
    property double newX: 0
    property double newY: 0

    //background
    Image{
        id: grad
        source: "./images/elec_back.png"
        anchors.fill: parent
    } 
    

    Timer{
        id: mouseTimer
        running: true
        repeat: true
        interval: 80

        onTriggered:{
            var point = hand.set_gest_data()
            var gest = hand.setMouse()
            var new_gest = gest[0]
            var old_gest = gest[1]

            console.log("Current Gest ", new_gest)
            console.log("Old Gest ", old_gest)
            console.log("Position ", point)

            //Case for mouse up after performing a mouse down
            if (new_gest == 0){
                hand.mouseRelease(window, Qt.LeftButton, Qt.NoModifier, point)
            }

            //Case for clicking
            else if (new_gest == 3) {
                hand.mouseClick(window, Qt.LeftButton, Qt.NoModifier, point)
            }
                
            //Case mouse down
            else if (new_gest == 1 && old_gest != 1) {
                hand.mousePress(window, Qt.LeftButton, Qt.NoModifier, point)
            }

            //Case for mouse down and drag
            else if (new_gest == 2 && old_gest != 2) {
                hand.mousePress(window, Qt.LeftButton, Qt.AltModifier, point)
            }
        }
    }

    /*Video{
        id: media
        source:  "./images/Background_slowmo.mkv"
        anchors.fill: parent
        loops: MediaPlayer.Infinite
        fillMode: VideoOutput.Stretch
        autoPlay: true
        Component.onCompleted:{
            focus = true
        }
        //flushMode: VideoOutput.FirstFrame
    }*/

    Covid{
        id: graph2
        width: 1200/1.5
        height: 550/1.5
    }

    Spotify{
        id: spotify_widget
        width: 1200/1.5
        height: 550/1.5
        //x: window.newX
        //y: window.newY
        Component.onCompleted:{
            x = window.newX 
            y = window.newY
            window.newX = spotify_widget.x + spotify_widget.width
            window.newY = spotify_widget.y
        }       
    }

    Gmail{
        id: gmail_widget
        width: 2000/2.5
        height: 1200/2.5
        Component.onCompleted:{
            x = window.newX 
            y = window.newY
            window.newX = gmail_widget.x + gmail_widget.width
            window.newY = gmail_widget.y
        }       
    }

   Weather{
        id: weather_widget
        width: 500
        height: 550
        Component.onCompleted:{
            x = window.newX 
            y = window.newY
            window.newX = 0
            window.newY = spotify_widget.y + spotify_widget.height
        }  
    }

    Signaling {
        id: calendar_widget
        width: 2000/2.5
        height: 1200/2.5
        Component.onCompleted:{
            x = window.newX 
            y = window.newY
        }  
    }

    Clockwindow{
        id: clock
        width: 2000/2.5
        height: 1200/2.5

        Binding on scaleVal{
            when: clock.activeFocus == true
            value: sizeAdjust.value
        }
    }

    MouseArea{
        id: mouseArea
        anchors.fill: parent
        propagateComposedEvents: true
        hoverEnabled: true

        onClicked:{
            mouse.accepted = false
        }

        onPressed:{
            mouse.accepted = false
        }

        onReleased:{
            mouse.accepted = false
        }
    }

    
    // Slider to adjust size of widgets
    /*Slider{
        id: sizeAdjust 
        // Positioning 
        anchors{
            right: window.right
            rightMargin: window.right/10
            bottom: window.bottom
            bottomMargin: window.bottom/10
        }
        //Size
        width: 500 
        height: 1000

        //Slider sizes
        from: 0
        to: 4
    }*/

    /*Button{
        x: parent.width-50
        y: 0
        width: 50
        height: 50
        palette.button: "#0e3066"
        
        Text{
            anchors.centerIn: parent
            text: "Exit"
            color: "white"
        }
        onClicked: Qt.quit()
    }*/
}