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

    /*background
    Image{
        id: grad
        source: "./images/elec_back.png"
        anchors.fill: parent
        z: -2
    }
    */
     Video{
        id: media
        source:  "./images/Background.mkv"
        anchors.fill: parent
        loops: MediaPlayer.Infinite
        fillMode: VideoOutput.Stretch
        autoPlay: true
        Component.onCompleted:{
            focus = true
        }
        //flushMode: VideoOutput.FirstFrame
    }
    //Icons
    RowLayout{
        id: iconrow
        anchors{
            top: parent.top
            topMargin: parent.height/1.4
            bottom: parent.bottom
            bottomMargin: parent.height/10
            left:parent.left
            leftMargin: parent.width/11
            }
        spacing: 2
         Butt{
            id: weather_Icon
            image: "./images/Weather_Icon.png"
            Layout.preferredWidth:250
            Layout.preferredHeight:200
            onTouched: {
                if weather_Icon.width == 0
            }
        }
         Butt{
            id: gmail_Icon
            image: "./images/Gmail_Icon.png"
            Layout.preferredWidth:250
            Layout.preferredHeight:200
        }
        Butt{
            id: calandar_Icon
            image: "./images/Calandar_Icon.png"
            Layout.preferredWidth:250
            Layout.preferredHeight:200
        }
        Butt{
            id: time_Icon
            image: "./images/Time_Icon.png"
            Layout.preferredWidth:250
            Layout.preferredHeight:200
        }
        Butt{
            id: corona_Icon
            image: "./images/Corona_Icon.png"
            Layout.preferredWidth:250
            Layout.preferredHeight:200
        }
        Butt{
            id: spotify_Icon
            image: "./images/Spotify_Icon.png"
            Layout.preferredWidth:250
            Layout.preferredHeight:200
        }

}
/*
    Timer{
        id: mouseTimer
        running: true
        repeat: true
        interval: 80
        onTriggered:{
            hand.set_gest_data()
            hand.setMouse()
        }
    }
*/


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
    /*
    Signaling {
        id: calendar_widget
        width: 2000/2.5
        height: 1200/2.5
        Component.onCompleted:{
            x = window.newX 
            y = window.newY
        }  
    }
    */
    Clockwindow{
        id: clock
        width: 2000/2.5
        height: 1200/2.5
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