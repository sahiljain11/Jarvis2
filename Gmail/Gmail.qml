import QtQuick 2.12
import QtQml.Models 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.0

import "../components"

// Initialize application window
JarvisWidget{

    //border
    Picture{
        image: "../images/Gmail_Frame.png"
        anchors.fill: parent
    }

    //Background
    Rectangle{
        id: back 
        color:"#00FFF5"
        anchors.fill: parent
        opacity: 0.4
    }

    //List of Email Previews
    ListView{

        id: emails
        width: parent.width/4
        height: parent.height
        clip: true
        focus: true

        anchors{
            left: parent.left
            leftMargin: parent.width/48
        }

        //Define model
        model: emailPreview

        //Gmail button delgate
        delegate: Component {
            id: emailDelegate
            Gmail_Butt{
                width: emails.width
                height: emails.height/3.5
                tit: "Subj: " + subject
                sen: "Sen.: " + sender
                snip: snippet
            }
        }
        
        spacing: parent.height/40
        
        ScrollBar.vertical: ScrollBar{
            id: scroll
            active : true
            height: emails.height
            width: emails.width/20
            minimumSize: 0.0
            policy: Qt.ScrollBarAlwaysOn
            stepSize: .005
            anchors.right: parent.right
        }
    }


    // List of labels
    ListView{
        id: labels
        height: parent.height/5
        width: parent.width/5
        spacing: 20

        anchors{
            right: parent.right
            rightMargin: parent.width/40
            top: parent.top
            topMargin: parent.height/2
            bottom: parent.bottom
            bottomMargin: parent.height/8
        }

        focus: true

        //define model
        model: labellist

        //Set delegate for label list
        delegate: Component {
            id: labelDelagate
            Gmail_Butt{
                width: labels.width
                height: labels.height/3
                tit: ''
                sen: ''
                snip: ''
                tit2: label
                halign: Text.AlignHCenter
                valign: Text.AlignVCenter
                fontSize: height/2.7
            }
        }

        ScrollBar.vertical:ScrollBar{
            id: labelscroll
            active : true
            height: labels.height
            width: labels.width/20
            minimumSize: 0.0
            policy: Qt.ScrollBarAlwaysOn
            stepSize: .005
            anchors.right: parent.right
        }
    }

    // List model for labels
    ListModel{
    id: labellist

        Component.onCompleted:{
            for (var i = 0; i < 5; i++) {
                labellist.append(label(i))
            }
        }

        function label(i) {

            return{
                label: gmail.get_label(i)
            };
        }
    }

    
    // Compose email button
    Butt{
        image: '../images/NewEmailIcon.png'

        width: parent.width/15
        height: parent.width/15
        anchors{
            right: searchbar.left
            rightMargin: searchbar.width/40
            top: parent.top
            topMargin: parent.height/25
        }
        onTouched: {
            compbox.vis= true
        }
    }

    //Email Box
    ComposeBox{
        id: compbox
        anchors{
            verticalCenter: parent.verticalCenter
            horizontalCenter: parent.horizontalCenter
        }

        width: parent.width/2.5
        height: parent.height/1.5
    }

    // Query bar for new emails
    TextField{
        id: searchbar
        anchors{
            top: parent.top
            topMargin: parent.height/20

            right: parent.right
            rightMargin: parent.width/40
            left: parent.left
            leftMargin: parent.width/1.5

        }
        Keys.onPressed:{
            if(event.key == Qt.Key_Return){
                gmail.get_messages_from_query(searchbar.text)
            }
        }
        height: parent.height/10

        //Styling
        color: 'black'
        font.pixelSize: searchbar.height/3
        /*background: Picture{
            anchors.fill: parent
            image: "../images/SearchBar.png"
        }*/
    }
}
