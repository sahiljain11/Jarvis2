import QtQuick 2.12
import QtQml.Models 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.0

import "../components"

// Initialize application window
ApplicationWindow{
    property alias vis: new_email.visible
    visible: true
    width: 2000
    height: 1200
    title: qsTr("Gmail")

   Image{
        source: '../images/background_gmail.png'
        height: parent.height
        width: parent.width
   }

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

   Item{
        id: gmail_container
        width: parent.width/2
        height: parent.height

        Component {
            id: emailDelegate
            Gmail_Butt{
                tit: subject
                sen: sender
                snip: snippet
            }
        }

        ListView{

            id: emails

            contentHeight: parent.height
            contentWidth: parent.width/2
            width: parent.width/2
            height: parent.height
            clip: true
            model: emailPreview
            delegate: emailDelegate
            spacing: parent.height/40
            Keys.onUpPressed: scroll.decrease()
            Keys.onDownPressed: scroll.increase()
            focus: true
            anchors{
                left: parent.left
                leftMargin: parent.width/24
            }
            
            ScrollBar.vertical:
                ScrollBar{
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
   }

   ListView{
        id: labels

        height: parent.height/5
        width: parent.width/5
        contentHeight: parent.height/4
        contentWidth: parent.width/5
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
        model: labellist
        delegate:labelDelagate
        ScrollBar.vertical:
           ScrollBar{
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

    Component {
            id: labelDelagate
            Gmail_Butt{
                tit: ''
                sen: label
                snip: ''

            }
    }

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

    Item{

        id: new_email
        anchors.fill: jarvis
        visible: false
        Rectangle{
            anchors.fill: new_email
            color: '#21ADE8'
            border.color: 'black'
            border.width: 5
            radius: 15
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
                TextEdit{
                    id: sendtotext
                    text: ""
                    cursorPosition: 5
                    anchors.fill: parent
                    wrapMode: TextEdit.Wrap


                }
            }
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
                TextEdit{
                    id: subjectStringText
                    text: ""
                    cursorPosition: 5
                    anchors.fill: parent
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
                   cursorPosition: 5
                   anchors.fill: parent
                   wrapMode: TextEdit.Wrap
                }
            }
        Label{
            color: 'blue'
            text: 'Recipient'
            font.pixelSize: 30
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
            font.pixelSize: 30
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
            font.pixelSize: 30
            anchors{
                top: subjectText.bottom
                bottom: bodyText.top
                left: bodyText.left
                right: bodyText.right
                rightMargin: bodyText.width/1.2
            }
        }
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

    }


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
    width: parent.width/10
    height: parent.height/10

    style: TextFieldStyle{
        textColor: 'black'
        font.pixelSize: 50
        background: Picture{
            image: "../images/SearchBar.png"
            }
        }
    }
   }

