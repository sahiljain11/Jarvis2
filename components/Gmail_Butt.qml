import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.11
import QtGraphicalEffects 1.12

Item{

    //Properties
    property bool vis: false
    property alias tit: title.text
    property alias sen: sender.text
    property alias snip: preview_email.text
    property alias tit2: title2.text
    property var fontSize: height/13
    property var halign: Text.AlignLeft
    property var valign: Text.AlignTop
    signal touched()
    signal inside()
    signal outside()

    FontLoader { id: nidsans; source: "../fonts/Nidsans.ttf"}
    FontLoader { id: astro; source: "../fonts/AstroSpace.ttf"}
    
    /*Rectangle{
        id: back
        anchors.fill: parent
        border.color: "#FFA54C"
        opacity: 0.7
    }*/

    Picture{
        id: back
        anchors.fill: parent
        image: "../images/Gmail_Border.png"
    }

    Text{
        id: title
        text: 'hello world'

        anchors{ 
            left: parent.left
            leftMargin: parent.width/15
            right: parent.right
            rightMargin: parent.width/18.2908 
            top : parent.top
            topMargin : parent.height/10
        }
        elide: Text.ElideRight
        font.pixelSize: fontSize
        font.family: nidsans.name
        color: "white"
        horizontalAlignment: halign
        verticalAlignment: valign
    }

    Text{
        id: sender
        text: 'conradiste@gmail.com'
        anchors{
            top: title.bottom
            topMargin: parent.height/20
            left: title.left
            right: title.right
            bottom: parent.bottom
            bottomMargin: parent.height/1.5
        }
        color: "white"
        elide: Text.ElideRight
        font.pixelSize: fontSize
        font.family: nidsans.name
        horizontalAlignment: halign
        verticalAlignment: valign
    }

    Text{
        id: title2
        anchors.fill: parent
        horizontalAlignment: halign
        verticalAlignment: valign
        font.pixelSize: fontSize
        font.family: nidsans.name
    }
    Text{
        id: preview_email
        clip : true
        anchors{
                left: title.left
                right: parent.right
                rightMargin: parent.width/18.2908
                top: sender.bottom
                topMargin: parent.height/20
                bottom: parent.bottom
                bottomMargin: parent.height/10
        }
        font.pixelSize: fontSize
        font.family: nidsans.name
        width: parent.width
        height: parent.height/2
        elide: Text.ElideRight
        wrapMode: Text.WordWrap
        text: "preview email new line af 33 charf jsdl kfja lsd jfwio  lwadfs dfajio wjd foawnefo nawd lfnhqo nwglnWIO PNWL HOIF OWIF HOWAGN Olj wodjfowan flwn gownag"
        horizontalAlignment: halign
        verticalAlignment: valign
    }

    Rectangle{
        id: mask
        color:'black'
        opacity: .3
        anchors.fill: parent
        visible: vis
        radius:5
    }
    //Darkens the button when the mouse is over it
    //Connects the clicked signal to 'touched'
    MouseArea{
        id: mou
        anchors.fill: parent
        hoverEnabled: true

        onEntered: {vis = true; entered.connect(inside);}

        onExited: {vis = false; exited.connect(outside)}

        onClicked: clicked.connect(touched)
    }
}