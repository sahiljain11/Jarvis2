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
    property int maxZ: 5
    property double iconRad: 49

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
        z: -20
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
            left: parent.left
            leftMargin: parent.width/9.8
            right: parent.right
            rightMargin: parent.width/7.7
        }
        
        spacing: 30

        Switch{
            id: weather_Icon
            iconOn: "./images/Weather_Icon.png"
            iconOff: "./images/Weather_Icon.png"
            Layout.preferredWidth:180
            Layout.preferredHeight:180
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            doTint: false
            rad: window.iconRad
            onTouched: {
                if (state == "ON"){
                   window.maxZ += 1
                   weather_widget.state = "opened"
                }
                else {
                    weather_widget.state = "closed"
                }
            }
        }
        
        Switch{
            id: gmail_Icon
            iconOn: "./images/Gmail_Icon.png"
            iconOff: "./images/Gmail_Icon.png"
            Layout.preferredWidth:180
            Layout.preferredHeight:180
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            doTint: false
            rad: window.iconRad
            onTouched: {
                if (state == "ON"){
                    window.maxZ += 1
                    gmail_widget.state = "opened"
                }
                else {
                    gmail_widget.state = "closed"
                }
            }
        }

        Switch{
            id: calandar_Icon
            iconOn: "./images/Calandar_Icon.png"
            iconOff: "./images/Calandar_Icon.png"
            Layout.preferredWidth:180
            Layout.preferredHeight:180
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            doTint: false
            rad: window.iconRad

            onTouched: {
                if (state == "ON"){
                    window.maxZ += 1
                    calendar_widget.state = "opened"
                }
                else {
                    calendar_widget.state = "closed"
                }
            }
        }

        Switch{
            id: time_Icon
            iconOn: "./images/Time_Icon.png"
            iconOff: "./images/Time_Icon.png"
            Layout.preferredWidth:180
            Layout.preferredHeight:180
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            doTint: false
            rad: window.iconRad
            onTouched: {
                if (state == "ON"){
                    window.maxZ += 1
                    clock_widget.state = "opened"
                }
                else {
                    clock_widget.state = "closed"
                }
            }
        }

        Switch{
            id: corona_Icon
            iconOn: "./images/Corona_Icon.png"
            iconOff: "./images/Corona_Icon.png"
            Layout.preferredWidth:180
            Layout.preferredHeight:180
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            doTint: false
            rad: window.iconRad
            onTouched: {
                if (state == "ON"){
                    window.maxZ += 1
                    covid_widget.state = "opened"
                }
                else {
                    covid_widget.state = "closed"
                }
            }
        }

        Switch{
            id: spotify_Icon
            iconOn: "./images/Spotify_Icon.png"
            iconOff: "./images/Spotify_Icon.png"
            Layout.preferredWidth:180
            Layout.preferredHeight:180
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            doTint: false
            rad: window.iconRad
            onTouched: {
                if (state == "ON"){
                    window.maxZ += 1
                    spotify_widget.state = "opened"
                }
                else {
                    spotify_widget.state = "closed"
                }
            }
        }
    }

    /*Timer{
        id: mouseTimer
        running: true
        repeat: true
        interval: 80
        onTriggered:{
            hand.set_gest_data()
            hand.setMouse()
        }
    }*/

    Covid{
        id: covid_widget
        width: 1200/1.5
        height: 550/1.5
        x: parent.width/2 - width/2
        y: parent.height/2.6 - height/2
    }

    Spotify{
        id: spotify_widget
        width: 1200/1.5
        height: 550/1.5
        x: parent.width/2 - width/2
        y: parent.height/2.6 - height/2
    }

    Gmail{
        id: gmail_widget
        width: 2000/2.5
        height: 1200/2.5
        x: parent.width/2 - width/2
        y: parent.height/2.6 - height/2
    }

    Weather{
        id: weather_widget
        width: 500
        height: 550
        x: parent.width/2 - width/2
        y: parent.height/2.6 - height/2
    }

    Signaling {
        id: calendar_widget
        width: 2000/2.5
        height: 1200/2.5
        x: parent.width/2 - width/2
        y: parent.height/2.6 - height/2
    }

    Clockwindow{
        id: clock_widget
        width: 400
        height: 100
        x: parent.width/2 - width/2
        y: parent.height/2.6 - height/2
    }
    

    
    /*MouseArea{
        id: mouseArea
        anchors.fill: parent
        propagateComposedEvents: true
        hoverEnabled: true

        onClicked:{
            mouse.accepted = false
        }

        onPressed:{
            console.log("pressed")
            mouse.accepted = false
        }

        onReleased:{
            mouse.accepted = false
        }
    }*/
    
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