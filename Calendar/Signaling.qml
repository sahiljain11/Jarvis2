import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Controls.Private 1.0
import QtQuick.Controls.Styles 1.4
import MyCalendar 1.0
import "../components"


JarvisWidget {

        Image{
            id: back
            source: "frame2.png"
            anchors.fill: parent
            smooth: true
            opacity: 1
            focus: true


            Rectangle{
                z: -2
                anchors.fill: parent
                color: "#00FFF5"
                opacity: 0.4
            }
        }

    SystemPalette {
        id: systemPalette
    }


    CalendarProvider {
        id: eventModel
        onLoaded: {
            // reload
            loader.sourceComponent = null
            loader.sourceComponent = page_component
        }
        // loads calendar events
        Component.onCompleted: {

            eventModel.updateListEvents({
                calendarId: "primary",
                timeMin: new Date(),
                maxResults: 20,
                singleEvents: true,
                orderBy: "startTime",
            })
        }
    }

    Loader{
        id: loader
        anchors.fill: parent
        sourceComponent: page_component
    }

    Component{
        id: page_component
        Flow {
            id: row
            anchors.fill: parent
            anchors.margins: 30
            spacing: 10
            layoutDirection: Qt.RightToLeft

            Calendar {
                id: calendar
                width: (parent.width > parent.height ? parent.width * 0.6 - parent.spacing : parent.width)
                height: (parent.height > parent.width ? parent.height * 0.6 - parent.spacing : parent.height)
                frameVisible: true
                selectedDate: new Date()
                focus: true
                style: CalendarStyle {
                    dayDelegate: Item {
                        readonly property color sameMonthDateTextColor: "#444"
                        readonly property color selectedDateColor: Qt.platform.os === "osx" ? "#282828" : systemPalette.highlight
                        readonly property color selectedDateTextColor: "white"
                        readonly property color differentMonthDateTextColor: "#bbb"
                        readonly property color invalidDatecolor: "#dddddd"

                        Rectangle {
                            anchors.fill: parent
                            border.color: "transparent"
                            color: styleData.date !== undefined && styleData.selected ? selectedDateColor : "transparent"
                            anchors.margins: styleData.selected ? -1 : 0
                        }

                        Image {
                            visible: eventModel.eventsForDate(styleData.date).length > 0
                            anchors.top: parent.top
                            anchors.left: parent.left
                            anchors.margins: -1
                            width: 12
                            height: width
                            source: "indicator.png"
                        }

                        Label {
                            id: dayDelegateText
                            text: styleData.date.getDate()
                            anchors.centerIn: parent
                            color: {
                                var color = invalidDatecolor;
                                if (styleData.valid) {
                                    // Date is within the valid range.
                                    color = styleData.visibleMonth ? sameMonthDateTextColor : differentMonthDateTextColor;
                                    if (styleData.selected) {
                                        color = selectedDateTextColor;
                                    }
                                }
                                color;
                            }
                        }
                    }
                }
            }

            // list of events for selected date
            Component {
                id: eventListHeader

                Row {
                    id: eventDateRow
                    width: parent.width
                    height: eventDayLabel.height
                    spacing: 10

                    Label {
                        id: eventDayLabel
                        text: calendar.selectedDate.getDate()
                        font.pointSize: 35
                    }

                    Column {
                        height: eventDayLabel.height

                        Label {
                            readonly property var options: { weekday: "long" }
                            text: Qt.locale().standaloneDayName(calendar.selectedDate.getDay(), Locale.LongFormat)
                            font.pointSize: 18
                        }
                        Label {
                            text: Qt.locale().standaloneMonthName(calendar.selectedDate.getMonth())
                                  + calendar.selectedDate.toLocaleDateString(Qt.locale(), " yyyy")
                            font.pointSize: 12
                        }
                    }
                }
            }

            Rectangle {
                width: (parent.width > parent.height ? parent.width * 0.4 - parent.spacing : parent.width)
                height: (parent.height > parent.width ? parent.height * 0.4 - parent.spacing : parent.height)
                border.color: Qt.darker(color, 1.2)
                anchors.margins: 30
                id: rectone
                ListView {
                    id: eventsListView
                    spacing: 4
                    clip: true
                    header: eventListHeader
                    anchors.fill: parent
                    anchors.margins: 10
                    model: eventModel.eventsForDate(calendar.selectedDate)

                    delegate: Rectangle {
                        width: eventsListView.width
                        height: eventItemColumn.height
                        anchors.horizontalCenter: parent.horizontalCenter

                        Image {
                            anchors.top: parent.top
                            anchors.topMargin: 4
                            width: 12
                            height: width
                            source: "indicator.png"
                        }

                        Rectangle {
                            width: parent.width
                            height: 1
                            color: "#eee"
                            opacity: .5
                        }

                        Column {
                            id: eventItemColumn
                            anchors.left: parent.left
                            anchors.leftMargin: 20
                            anchors.right: parent.right
                            height: timeLabel.height + nameLabel.height + 8

                            Label {
                                id: nameLabel
                                width: parent.width
                                wrapMode: Text.Wrap
                                text: modelData["summary"]
                            }
                            Label {
                                id: timeLabel
                                width: parent.width
                                wrapMode: Text.Wrap
                                text: modelData.start.toLocaleTimeString(calendar.locale, Locale.ShortFormat) + "-" + modelData.end.toLocaleTimeString(calendar.locale, Locale.ShortFormat)
                                color: "#aaa"
                            }
                        }
                    }

                    // creating an event from the calendar
                    TextField {
                        id: eventstart
                        anchors.bottom: eventend.top
                        anchors.right: parent.right
                        anchors.left: parent.left
                        placeholderText: qsTr("Start Time 01/12/2020 14:35:00")
                        selectByMouse: true
                    }
                    TextField {
                        id: eventend
                        anchors.bottom: eventinfo.top
                        anchors.right: parent.right
                        anchors.left: parent.left
                        placeholderText: qsTr("End Time 01/12/2020 16:35:00")
                        selectByMouse: true
                    }
                    TextField {
                        id: eventinfo
                        anchors.bottom: buttonn.top
                        anchors.right: parent.right
                        anchors.left: parent.left
                        placeholderText: qsTr("Event Name")
                        selectByMouse: true
                    }
                    Butt{
                        id: buttonn
                        width: rectone.width
                        anchors.bottom: parent.bottom
                        anchors.right: parent.right
                        anchors.left: parent.left
                        height: rect.height

                        Rectangle {
                            id: rect
                            implicitWidth: rectone.width
                            implicitHeight: 25
                            color: "#282828"
                            radius: 10
                        }

                        Text {
                            text: "Add Event"
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.verticalCenter: parent.verticalCenter
                            color: "white"
                        }
                        onTouched: {
                            var dt_start = Date.fromLocaleString(Qt.locale(), eventstart.text, "MM/dd/yyyy hh:mm:ss")
                            var dt_end = Date.fromLocaleString(Qt.locale(), eventend.text, "MM/dd/yyyy hh:mm:ss")
                            if(dt_start.getDate() && dt_end.getDate()){
                                eventModel.createEvent({
                                    calendarId: "primary",
                                    body: {
                                        summary: eventinfo.text,
                                        start: {
                                            dateTime: dt_start,
                                            timeZone: "America/Chicago",
                                        },
                                        end: {
                                            dateTime: dt_end,
                                            timeZone: "America/Chicago",
                                        },
                                    }
                                })
                            }
                            //reloads loader,

                            //eventModel.updateListEvents({
                            //calendarId: "primary",
                            //timeMin: new Date(),
                            //maxResults: 20,
                            //singleEvents: true,
                            //orderBy: "startTime",
                        //})
                            //loader.active = !loader.active

                            //loader.active = !loader.active

                        }
                    }



                }
            }
        }
    }
}



