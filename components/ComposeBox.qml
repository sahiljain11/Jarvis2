import QtQuick 2.12
import QtQml.Models 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.0

import "."


Rectangle{
    id: new_email
    color: '#42d7f5'
    border.color: 'blue'
    //border.width: 5
    //radius: 15

    property alias vis: new_email.visible

    visible: false

    // Receiver
    Rectangle{
        id: sendto
        anchors{
            top: new_email.top
            topMargin: new_email.height/15
            left: new_email.left
            leftMargin: new_email.width/15
            right: new_email.right
            rightMargin: new_email.width/15
            bottom: new_email.bottom
            bottomMargin: new_email.height/1.2
        }
        color: 'white'
        opacity: 0.5
        //radius: 15
    }

    // Input Receiver
    TextEdit{
        id: sendtotext
        anchors{
            left: sendto.left
            leftMargin: sendto.width/30
            top: sendto.top
            topMargin: sendto.height/10
            bottom: sendto.bottom
            bottomMargin: sendto.height/50
            right: sendto.right
            rightMargin: sendto.width/30
        }
        wrapMode: TextEdit.Wrap
    }

    //Subject Box
    Rectangle{
        id: subjectText
        color: 'white'
        //radius: 15
        opacity: 0.5
        layer.enabled: true
        anchors{
            top: sendto.bottom
            topMargin: sendto.height/2
            bottom: new_email.bottom
            bottomMargin: new_email.height/1.4
            left: sendto.left
            right: sendto.right
        }
    }

    // Subject input
    TextEdit{
        id: subjectStringText
        text: ""
        anchors{
            left: subjectText.left
            leftMargin: subjectText.width/30
            top: subjectText.top
            topMargin: subjectText.height/9
            bottom: subjectText.bottom
            bottomMargin: subjectText.height/50
            right: subjectText.right
            rightMargin: subjectText.width/30
        }
        wrapMode: TextEdit.Wrap
    }

    Rectangle{
        id: bodyText
        color: 'white'
        //radius: 15
        opacity: 0.5
        anchors{
            top: subjectText.bottom
            topMargin: subjectText.height/1.5
            bottom: new_email.bottom
            bottomMargin: new_email.height/5
            left: subjectText.left
            right: subjectText.right
        }
    }

    TextEdit{
        id: bodyStringText
        text: ""
        anchors{
            left: bodyText.left
            leftMargin: bodyText.width/35
            top: bodyText.top
            topMargin: bodyText.height/30
            bottom: bodyText.bottom
            bottomMargin: bodyText.height/35
            right: bodyText.right
            rightMargin: bodyText.width/35
        }
        wrapMode: TextEdit.Wrap
    }

    // Labels for text boxes
    Label{
        color: 'blue'
        text: 'Recipient'
        fontSizeMode: Text.Fit
        anchors{
            top: new_email.top
            topMargin: new_email.height/40
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
            top: new_email.top
            topMargin: new_email.height/1.2
            bottom: new_email.bottom
            bottomMargin: new_email.height/30
            right: new_email.right
            rightMargin: new_email.width/25
            left: new_email.left
            leftMargin: new_email.width/2
        }

        onTouched:{

            gmail.send_message("me", sendtotext.text, subjectStringText.text, bodyStringText.text)

            new_email.visible = false
            sendtotext.text=""
            subjectStringText.text=""
            bodyStringText.text= ""
        }
    }
}
