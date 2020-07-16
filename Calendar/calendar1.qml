
import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.2

ApplicationWindow {
    title: qsTr("Calendar")
    width: 500
    height: 400
    visible: true


    Calendar {
    weekNumbersVisible: true
}
}