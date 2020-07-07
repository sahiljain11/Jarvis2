import QtQuick 2.5
import QtQuick.Controls 2.12
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
    title: qsTr("Spotify")

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

        //play/pause switch
        Switch{
            id: play

            //Load the play button image
            iconOn: "./images/pause.png"
            iconOff: "./images/play.png" 

            //Position the play button at the bottom center
            anchors {
                horizontalCenter: parent.horizontalCenter
                bottom: parent.bottom
                bottomMargin: parent.height/20
            }

            onStateChanged: {
                if (state == "ON"){
                    spotify.play_music()
                }
                else{
                    spotify.pause_music()
                }
            }

            //Set the size
            width: picHeight/2
            height: picWidth/2
        }


        // Skip_forward button
        Butt{
            id: skip_forward
            
            //Load the skip png
            image: "./images/skip_forward.png"

            //Position the skip button to the right of the skip button
            anchors{
                bottom: play.bottom
                verticalCenter: play.verticalCenter
                left: play.right
                leftMargin: parent.width/15
            }

            onTouched: {
                console.log("forward")
                spotify.next_song()
                play.state = 'ON'
                song_timer.restart()
            }

            //Resize
            width: picWidth/3
            height: picHeight/3
        }

        //Skip backward button
        Butt{
            id: skip_backward
            
            //Load the skip png
            image: "./images/skip_back.png"

            //Position the skip button to the right of the skip button
            anchors{
                bottom: play.bottom
                verticalCenter: play.verticalCenter
                right: play.left
                rightMargin: parent.width/15
            }
            
            onTouched: {
                console.log("back")
                spotify.prev_song()
                play.state = 'ON'
                song_timer.restart()
            }
            
            //Resize
            width: picWidth/3
            height: picHeight/3
        }

        //Artist
        Text{
            id: artist
            anchors{
                horizontalCenter: play.horizontalCenter
                bottom: play.top
                bottomMargin: play.height/1.5
            }
            //font.bold: true
            font.pointSize: 8
            font.family: nidsans.name
            
            //Style
            color: "white"
            
            text: spotify.currArtist
        }

        //Song Title
        Text{
            id: title
            anchors{
                horizontalCenter: artist.horizontalCenter
                bottom: artist.top
                bottomMargin: play.height/2
            }

            //font.bold: true
            font.pointSize: 9
            font.family: nidsans.name
            
            //Style
            color: "white"
            text: spotify.currTitle
        }

        //Song icon
        Image{
            id: song_icon

            //Set the song icon
            source: spotify.currIcon
            // Position the song icon right above the play button
            // Make the song icon width bound by the skip buttons
            anchors{
                bottom: title.top
                bottomMargin: parent.width/40
                left: skip_backward.left
                right: skip_forward.right
            }   

            //Make the song icon square
            height: width
        }

        Timer{
            id: song_timer
            running: true
            repeat: true
            interval: 500
            onTriggered:{
                song_playback.value = spotify.get_current_time()
                if(song_playback.value > 0 && song_playback.value <= 600 ){
                    spotify.set_current_song_info()
                }
            }
        }

        //Dial that changes current song playback
        SongPlayBack{
            id: song_playback
            from: 0
            to: spotify.durTime
            stepSize: 1000

            anchors{
                left: parent.left
                leftMargin: parent.width/10
                right: skip_backward.right
                rightMargin: parent.width/10
                top: song_icon.verticalCenter
                bottom: parent.bottom
                bottomMargin: parent.height/10
            }
            
            onMoved:{
                spotify.change_time(song_playback.value)
            }
        }

        //Create the border
        Rectangle{
            
            //Positioning
            anchors{
                left: parent.left
                leftMargin: parent.width/18
                right: song_icon.left
                rightMargin: parent.width/30
                bottom: song_icon.verticalCenter
                bottomMargin: parent.height/16
                top: parent.top
                topMargin: parent.height/8
            }

            color: "transparent"
            
           //Testing Purposes to see the box
           //border{
                //width: 1
                //color: "#00FFF5"
           //}
            
            //First Line
            Text{
                //Positioning
                anchors{
                    top: parent.top
                    topMargin: parent.height/6
                    bottom: parent.verticalCenter
                    left: parent.left
                    right: parent.right
                }

                //Font settings
                fontSizeMode: Text.Fit
                font.family: astro.name
                font.pixelSize: parent.height/13

                //Style
                color: "white"
                
                //Align text in the center
                horizontalAlignment: Text.AlignHCenter

                //Wrap the text
                wrapMode: Text.WordWrap
                text: "Is this the real life? Is this just fantasy?"
            }

            //Second Line
            Text{
                anchors{
                    top: parent.verticalCenter
                    topMargin: parent.height/16
                    bottom: parent.bottom
                    bottomMargin: parent.height/8
                    left: parent.left
                    right: parent.right
                }

                //Font Settings 
                fontSizeMode: Text.Fit
                font.family: astro.name
                font.pixelSize: parent.height/13

                //Style
                color: "white"
                
                //Align text in the center
                horizontalAlignment: Text.AlignHCenter
                
                wrapMode: Text.WordWrap
                text: "Is this the real life? Is this just fantasy"
            }
        }


        //Volume
        Volume{
            id: volume
            anchors {
                
                //For vertical
                //left: skip_forward.right
                //leftMargin: parent.width/25
                //bottom: skip_forward.bottom
                //top: song_icon.bottom 
                
                //For horizontal
                left: skip_forward.right
                leftMargin: parent.width/25
                bottom: skip_forward.bottom
                top: skip_forward.top
            }

            width: parent.width/10
            
            onValueChanged: {
                spotify.change_volume(volume.value)
            }

            //Vertical Rotation
            //transform: Rotation {origin.x: x; origin.y: y; angle: -90}
        }

        //Text input for spotify
        TextField{
            id: textInput
            text: "Text"
            anchors{
                top: parent.top
                topMargin: parent.height/8
                right: parent.right
                rightMargin: parent.width/10
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

            if(event.key == Qt.Key_Enter){
                console.log("Prese")
                spotify.add_song_to_queue(textInput.text)
                spotify.helper_add_song_to_queue(0)
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