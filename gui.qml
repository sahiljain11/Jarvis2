import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12
import QtMultimedia 5.12
import QtQml.Models 2.12
import "./components"
import "./Spotify"
import "./Gmail"
import "./Weather"
import "./Calendar"

ApplicationWindow{
    id: window
    visible: true
    width: 600
    height: 600
    title: qsTr("Jarvis2")
    property double newX: 0
    property double newY: 0

    //background
    Image{
        id: grad
        source: "./images/elec_back.png"
        anchors.fill: parent
    }

    /*Video{
        id: media
        source:  "./images/FirstVideo.mkv"
        anchors.fill: parent
        loops: MediaPlayer.Infinite
        fillMode: VideoOutput.Stretch
        autoPlay: true
    }*/

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
        width: 2000/2.5
        height: 1200/2.5
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
    
}