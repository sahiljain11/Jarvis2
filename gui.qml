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
    visible: true
    width: 600
    height: 600
    title: qsTr("Spotify")

    //background
    Image{
        id: grad
        source: "./images/grad.png"
        anchors.fill: parent
    }

    //Video{
        //id: media
        //source:  "../FirstVideo.mkv"
        //anchors.fill: parent
        //loops: MediaPlayer.Infinite
        //fillMode: VideoOutput.Stretch
        //autoPlay: true
    //}

    Spotify{
        id: spotify_widget
        width: 1200/1.5
        height: 550/1.5
    }

    Gmail{
        id: gmail_widget
        x: spotify_widget.x+spotify_widget.width
        y: spotify_widget.height
        width: 2000/2.5
        height: 1200/2.5
    }
    
}