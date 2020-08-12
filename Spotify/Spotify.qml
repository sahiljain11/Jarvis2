import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12
import QtMultimedia 5.12
import QtQml.Models 2.12
import "../components"

JarvisWidget{
    Item {
        id: spotify_widget
        
        //Load fonts
        FontLoader { id: nidsans; source: "../fonts/Nidsans.ttf"}
        FontLoader { id: astro; source: "../fonts/AstroSpace.ttf"}
        
        //positioning
        width: parent.width
        height: parent.height

        // Sets the background image
        Picture{
            id: back
            image: "../images/frame1.png"
            anchors.fill: parent
            smooth: true
            opacity: 1
            focus: true
            //tint: "white"

            Rectangle{
                z: -2
                anchors.fill: parent
                color: "#00FFF5"
                opacity: 0.4
            }
        }

        //play/pause switch
        Switch{
            id: play

            //Load the play button image
            iconOn: "../images/pause.png"
            iconOff: "../images/play.png" 

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
            image: "../images/skip_forward.png"

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
            image: "../images/skip_back.png"

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
            interval: 1000
            onTriggered:{
                song_playback.value = spotify.get_current_time()
                if(song_playback.value <= 3000){
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
                leftMargin: parent.width/20
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

        //Search bar
        TextField{
            id: textInput
            anchors{
                top: parent.top
                topMargin: parent.height/8
                right: parent.right
                rightMargin: parent.width/20
            }
            font.family: astro.name
            scale: Math.min(1, parent.width / contentWidth)
        }


        //Toggle between searching for playlist and song
        Switch{
            id: toggle

            //Positioning
            anchors{
                left: textInput.left
                leftMargin: textInput.width/8
                bottom: textInput.top
                top: parent.top
                topMargin: parent.height/15
            }
            width: textInput.width/2.5

            //Set the default parameters
            word: "Search Songs"
            iconOff: "../images/back2.png"
            iconOn: "../images/back2.png"

            //Toggle functionality
            onTouched:{
                if (toggle.state == "ON"){
                    toggle.word = "Search Playlists"
                }
                else{
                    toggle.word = "Search Songs"
                }
            }
        }

        Component{
            id:search_del
            Text{
                anchors.fill: parent
                text: song
            }
        }

        //Search results
        ListView{
            id: search_results
            anchors{
                top: textInput.bottom
                bottom: volume.top
                bottomMargin: parent.height/20
                left: textInput.left
                right: textInput.right
            }
            model: searchList
            interactive: false
            delegate: Item {
                width: search_results.width
                height: search_results.height/10
                property bool vis: false

                Rectangle{
                    anchors.fill: parent
                    color: "#25b3e6"
                    opacity: 0.5
                }

                Text{
                    anchors.fill: parent
                    elide: Text.ElideRight
                    text: model.song + " by " + model.artist
                    fontSizeMode: Text.Fit
                }

                Rectangle{
                    anchors.fill: parent
                    visible: vis
                    color: "black"
                    opacity: 0.3
                }

                MouseArea{
                    anchors.fill: parent
                    hoverEnabled: true

                    onClicked:{
                        textInput.remove(0, textInput.text.length)
                        
                        //We are choosing a song
                        if (toggle.state == "OFF"){
                            spotify.helper_add_song_to_queue(index)
                        }

                        // We are choosing a playlist
                        else{
                            spotify.queue_music_from_playlist(index)
                        }
                    }

                    onEntered:{
                        vis = true
                    }

                    onExited:{
                        vis = false
                    }
                }
            }
        }

        //Defines key event changes
        Keys.onPressed: {
            if(event.key == Qt.Key_Return){

                if (toggle.state == "OFF"){
                    spotify.add_song_to_queue(textInput.text)
                }

                else{
                    spotify.find_a_playlist(textInput.text)
                }
            }
        }
    }
}