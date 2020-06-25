import QtQuick 2.6
import QtQuick 2.12
import QtQml.Models 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.0

import "./components"

// Initialize application window
ApplicationWindow{
    visible: true
    width: 2000
    height: 1200
    title: qsTr("Gmail")

   Image{
        source: 'images/background_gmail.png'
        height: parent.height
        width: parent.width
   }

   Image{
        source: 'images/Jarvis_Placeholder_1.png'

        anchors{
        verticalCenter: parent.verticalCenter
        horizontalCenter: parent.horizontalCenter
        }
        width: parent.width/5
        height: parent.height/3
   }

   Item{
        id: gmail_container
        width: parent.width/2
        height: parent.height


        ObjectModel {
            id: itemModel
            Gmail_Butt { height: list.height/4; width: list.width;  }
            Gmail_Butt { height: list.height/4; width: list.width; color: "green" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "blue" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "white" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "red" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "green" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "blue" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "white" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "red" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "green" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "blue" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "white" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "red" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "green" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "blue" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "white" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "red" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "green" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "blue" }
            Gmail_Butt { height: list.height/4; width: list.width; color: "white" }




        }
        ListView{

        id: list

        contentHeight: parent.height
        contentWidth: parent.width/2
        width: parent.width/2
        height: parent.height
        clip: true
        model: itemModel
        spacing: parent.height/40
        Keys.onUpPressed: scroll.decrease()
        Keys.onDownPressed: scroll.increase()
        focus: true
        anchors{
            left: parent.left
            leftMargin: parent.width/12
        }
        ScrollBar.vertical:
            ScrollBar{
            id: scroll
            active : true
            height: list.height
            width: list.width/20
            minimumSize: 0.0
            policy: Qt.ScrollBarAlwaysOn
            stepSize: .005


            anchors.right: parent.right
            }



        }

        }
   }

