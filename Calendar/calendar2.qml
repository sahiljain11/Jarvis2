import QtQuick.Controls 1.2
import QtQuick 2.3
import QtQuick.Controls.Styles 1.2
import QtQuick.Layouts 1.11

ApplicationWindow {
    title: qsTr("Calendar")
    width: 700
    height: 400
    visible: true


    RowLayout {
    anchors.fill: parent

        Rectangle {
        id: tasks
        Layout.fillWidth: true
        Layout.fillHeight: true
        height: 400
        width: 150
        color: "white"
            ListView{
            model: Cal
            anchors.fill: parent
            delegate: Text { text: cal.display }
        }
        Component.onCompleted: {
            var data = cal.getevents()
            for(var key in data){
                var value = data[key]
                console.log(key, ": ", value)
            }
        }


        }
        Calendar {
        Layout.fillWidth: true
        Layout.fillHeight: true
        frameVisible: true
        style: CalendarStyle {
            gridVisible: false
            dayDelegate: Rectangle {
                gradient: Gradient {
                    GradientStop {
                        position: 0.00
                        color: styleData.selected ? "#111" : (styleData.visibleMonth && styleData.valid ? "#444" : "#666");
                    }
                    GradientStop {
                        position: 1.00
                        color: styleData.selected ? "#444" : (styleData.visibleMonth && styleData.valid ? "#111" : "#666");
                    }
                    GradientStop {
                        position: 1.00
                        color: styleData.selected ? "#444" : (styleData.visibleMonth && styleData.valid ? "#111" : "#666");
                    }
                }

                Label {
                    text: styleData.date.getDate()
                    anchors.centerIn: parent
                    color: styleData.valid ? "white" : "white"
                }

                Rectangle {
                    width: parent.width
                    height: 1
                    color: "#555"
                    anchors.bottom: parent.bottom
                }

                Rectangle {
                    width: 1
                    height: parent.height
                    color: "#555"
                    anchors.right: parent.right
                }
            }
        }
        }
    }
}