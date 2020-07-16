import QtQuick 2.12
import QtQml.Models 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.0

import "../components"

// Initialize application window
Image{
    property alias vis: new_email.visible
    source: '../images/background_gmail.png'
    visible: true
    width: 2000
    height: 1200

   Image{
        id: jarvis
        source: '../images/Jarvis_Placeholder_1.png'
        anchors{
            verticalCenter: parent.verticalCenter
            horizontalCenter: parent.horizontalCenter
        }
        width: parent.width/2.5
        height: parent.height/1.5
    }

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
                height: emails.height/10
                tit: subject
                sen: sender
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
            bottomMargin: parent.height/24
        }

        focus: true

        //define model
        model: labellist

        //Set delegate for label list
        delegate: Component {
            id: labelDelagate
            Gmail_Butt{
                width: labels.width
                height: labels.height/5
                tit: ''
                sen: label
                snip: ''
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
        onTouched: {vis= true}
    }

    //Email Box
    Rectangle{
        id: new_email
        color: '#21ADE8'
        border.color: 'black'
        border.width: 5
        radius: 15
        visible: false

        anchors.fill: jarvis

        // Receiver
        Rectangle{
            id: sendto
            anchors{
                top: parent.top
                topMargin: parent.height/15
                left: parent.left
                leftMargin: parent.width/15
                right: parent.right
                rightMargin: parent.width/15
                bottom: parent.bottom
                bottomMargin: parent.height/1.2
            }
            color: 'white'

            radius: 15

            // Input Receiver
            TextEdit{
                id: sendtotext
                anchors{
                    left: parent.left
                    leftMargin: parent.width/30
                    top: parent.top
                    topMargin: parent.height/10
                    bottom: parent.bottom
                    bottomMargin: parent.height/50
                    right: parent.right
                    rightMargin: parent.width/30
                }
                wrapMode: TextEdit.Wrap
            }
        }

        //Subject Box
        Rectangle{
            id: subjectText
            color: 'white'
            radius: 15

            anchors{
                top: sendto.bottom
                topMargin: sendto.height/2
                bottom: parent.bottom
                bottomMargin: parent.height/1.4
                left: sendto.left
                right: sendto.right
            }

            // Subject input
            TextEdit{
                id: subjectStringText
                text: ""
                anchors{
                    left: parent.left
                    leftMargin: parent.width/30
                    top: parent.top
                    topMargin: parent.height/9
                    bottom: parent.bottom
                    bottomMargin: parent.height/50
                    right: parent.right
                    rightMargin: parent.width/30
                }
                wrapMode: TextEdit.Wrap
            }
        }

        Rectangle{
            id: bodyText
            color: 'white'
            radius: 15
            anchors{
                top: subjectText.bottom
                topMargin: subjectText.height/1.5
                bottom: parent.bottom
                bottomMargin: parent.height/5
                left: subjectText.left
                right: subjectText.right
            }
            TextEdit{
                id: bodyStringText
                text: ""
                anchors{
                    left: parent.left
                    leftMargin: parent.width/35
                    top: parent.top
                    topMargin: parent.height/30
                    bottom: parent.bottom
                    bottomMargin: parent.height/35
                    right: parent.right
                    rightMargin: parent.width/35
                }
                wrapMode: TextEdit.Wrap
            }
        }

        // Labels for text boxes
        Label{
            color: 'blue'
            text: 'Recipient'
            fontSizeMode: Text.Fit
            anchors{
                top: parent.top
                topMargin: parent.height/40
                bottom: sendto.top
                left: sendto.left
                right: sendto.right
                rightMargin: sendto.width/1.2
            }
        }

        Label{
            color: 'blue'
            text: 'Subject'
            fontSizeMode: Text.Fit
            anchors{
                top: sendto.bottom
                bottom: subjectText.top
                left: subjectText.left
                right: subjectText.right
                rightMargin: subjectText.width/1.2
            }
        }
        
        Label{
            color: 'blue'
            text: 'Message'
            fontSizeMode: Text.Fit
            anchors{
                top: subjectText.bottom
                bottom: bodyText.top
                left: bodyText.left
                right: bodyText.right
                rightMargin: bodyText.width/1.2
            }
        }

        // Send email button
        Butt{
            id: send_butt

            image: '../images/SendEmailIcon.png'
            anchors{
                top: parent.top
                topMargin: parent.height/1.2
                bottom: parent.bottom
                bottomMargin: parent.height/30
                right: parent.right
                rightMargin: parent.width/25
                left: parent.left
                leftMargin: parent.width/2
            }
            onTouched:{

                gmail.send_message("me", sendtotext.text, subjectStringText.text, bodyStringText.text)

                vis = false
                sendtotext.text=""
                subjectStringText.text=""
                bodyStringText.text= ""
            }
        }
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
            if(event.key == Qt.Key_Left){
                gmail.get_messages_from_query(searchbar.text)
            }
        }
        height: parent.height/10

        //Styling
        color: 'black'
        font.pixelSize: searchbar.height/3
        background: Picture{
            anchors.fill: parent
            image: "../images/SearchBar.png"
        }
    }
}

