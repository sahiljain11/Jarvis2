import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Templates 2.0 as T
import QtGraphicalEffects 1.12
import "."



Dial{
    id: song_playback
    from: 0
    to: 10000
    value: 0
    live: true
    
    palette.dark: 'white'


    //implicitWidth: 200
    //implicitHeight: 200

    Text{
        text: msToTime(Math.round(value))
        anchors.centerIn: parent
        color: "white"
    }

    //Takes in a time in milliseconds and returns a sting representation of the time
    function msToTime(ms){
        var seconds = ms/1000
        var minutes = Math.floor(seconds/60)
        var seconds = Math.floor(seconds % 60)
        if(String(+seconds).charAt(0) == seconds){
            seconds = "0" + seconds
        }
        return  "" + minutes + ":" + seconds
    }

    

}