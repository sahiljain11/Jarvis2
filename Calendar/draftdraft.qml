import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Controls.Private 1.0
import QtQuick.Controls.Styles 1.4

ApplicationWindow {
    visible: true
    width: 640
    height: 400
    minimumWidth: 400
    minimumHeight: 300


    title: "My Calendar"

    Component{
        id: page_component


            Calendar {

                id: calendar
                width: (parent.width > parent.height ? parent.width * 0.6 - parent.spacing : parent.width)
                height: parent.height * .6
                frameVisible: true
                anchors.top: parent.top
                selectedDate: new Date()
                focus: true
            }

            Rectangle {
                width: (parent.width > parent.height ? parent.width * 0.4 - parent.spacing : parent.width)
                height: parent.height * .6
                anchors.margins: 30
                anchors.top: parent.top
                color: blue

            }
            Rectangle {
                width: 640
                height: parent.height * .4
                anchors.bottom: parent.bottom
                color: red
            }

        }


}
