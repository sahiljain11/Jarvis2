import QtQuick 2.7
import QtQuick.Templates 2.0 as T
import QtGraphicalEffects 1.12
import "."

T.Slider {
    id: control

    implicitWidth: 200
    implicitHeight: 26
    from: 0
    to: 100

    //Set is the inital value of the volume bar
    property alias initialVal: control.value
    initialVal: to/2
    
    //Make the handle invisible
    handle: Rectangle {
        x: control.visualPosition * (control.width - width)
        y: (control.height - height) / 2
        width: parent.width/20
        height: parent.height
        color: "black"
        visible: false
    }


    //Set the background image to the volume bar
    //Pull a gray rectangle over the volume bar as the handle is moved
    background: Item{

        Image{
            id: sliderImage
            source: "../images/volume_barH.png"
            height: parent.height*.95
            width: parent.width*.95
            anchors.centerIn: parent
        }

        Rectangle{
            id: outer
            anchors{
                top: parent.top
                bottom: parent.bottom
                right: parent.right
            }
            width:(1-control.visualPosition)*parent.width
            color: "#00FFF5"
            opacity: 0.95
            radius: 1.5
        }
    }
}