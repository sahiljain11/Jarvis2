import QtQuick 2.12
import QtQml.Models 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtMultimedia 5.0
import GmailMod 1.0

import "../components"

// Initialize application window
JarvisWidget{

    //Boolean to tell if we are replying
    property bool replying: false
    property bool justStarted: true

    GmailModule {
        //Initializes the gmail widget
        Component.onCompleted:{
            //Retrieve the first 50 emails from the all mail inbox
            gmail.init_gmail_in_widget(50)
            //Add the labels
            for (var i = 0; i < 5; i++) {
                labellist.append({label: gmail.get_label(i)})
            }
        }
    }

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

        onCurrentItemChanged:{
            threads.scale = 1
            gmail.add_thread_messages(emails.currentIndex)

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

                onTouched:{
                    emails.currentIndex = index
                }
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

    ListView {
        id: threads
        anchors{
            verticalCenter: parent.verticalCenter
            horizontalCenter: parent.horizontalCenter
        }
        
        //Orients the list model side ways 
        snapMode: ListView.SnapOneItem
        highlightRangeMode: ListView.StrictlyEnforceRange
        highlightMoveDuration: 250
        orientation: ListView.Horizontal
        boundsBehavior: Flickable.StopAtBounds
        clip: true

        model: threadMessages

        //Gmail button delgate
        delegate: Component {
            id: thread_delegate
            Gmail_Msg{
                width: threads.width
                height: threads.height
                msg: message
            }
        }

        width: parent.width/2.5
        height: parent.height/1.5
    }

    //Replying button
    Butt{
        id: reply
        image: "../images/replyArrow.png"
        //Positioning 
        anchors{
            top: threads.bottom
            topMargin: parent.height/50
            right: threads.right
        }
        width: 50
        height: 50
        scale: threads.scale

        onTouched:{

            //Reduce the message disaply to 0
            threads.scale = 0
            var index = emails.currentIndex

            //Grab the current subject, sender, and threadid
            compbox.subj = gmail.get_current_subject(index)
            compbox.send = gmail.get_current_sender(index)
            compbox.threadid = gmail.get_current_threadid(index)
            compbox.reply_msgid = gmail.get_current_message_id(index)
            compbox.references = gmail.get_current_references(index)
            compbox.readOnlySubj = true
            compbox.readOnlySend = true
            console.log(compbox.references)
            console.log(compbox.subj)

            //We are now replying to the same thread
            compbox.replying = true
            compbox.vis = true
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

        onCurrentItemChanged:{
            console.log("The current item changed")
            
            //Avoids the issue where the current item changes automatically when the app opens
            if(!parent.justStarted){
                gmail.get_messages_with_labels(gmail.get_label(labels.currentIndex))
            }
            else{
                parent.justStarted = false
            }
        }

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
                
                onTouched:{
                    labels.currentIndex = index
                }
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

    }

    
    // New email button
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
            compbox.vis = !compbox.vis
            compbox.subj = ""
            compbox.send = ""
            compbox.body = ""
            compbox.readOnlySubj = false
            compbox.readOnlySend = false
            if (compbox.vis == true){
                compbox.replying = true
                threads.scale = 0
            }
            else{
                threads.scale = 1
            }
        }
    }

    //Email Box
    ComposeBox{
        id: compbox
        anchors{
            verticalCenter: parent.verticalCenter
            horizontalCenter: parent.horizontalCenter
        }

        property string threadid: ""
        property string reply_msgid: ""
        property string references: ""

        width: parent.width/2.5
        height: parent.height/1.5

        onBumped:{
            console.log("Bumped")
            threads.scale = 1
            compbox.readOnlySubj = false
            compbox.readOnlySend = false
            // We are writing a new message
            if (!replying) {
                gmail.send_message("me", send, subj, body)
            }
            // We are responding to the current thread
            else {
                gmail.respond_to_thread(threadid, "me", send, subj, reply_msgid, references, body)
                send = ""
                subj = ""
                body = ""
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
