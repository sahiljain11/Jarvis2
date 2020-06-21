import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Styles 1.4
import QtQml 2.0
import QtGraphicalEffects 1.12
import QtQuick.Layouts 1.15

ApplicationWindow {
    title: qsTr("Weather App")
    width: 500
    height: 300
    visible: true
        Image {
            source: "default2.png"
            id: image_tod
            anchors.fill: parent
        }
    FontLoader {
        id: webFont
        source: "Orbitron-Medium.ttf"}


    ColumnLayout {
        anchors.fill: parent
        spacing: 20
        anchors.top: parent.top
        TextField {
            id: city_tf
            Layout.topMargin: 30
            placeholderText: qsTr("City")
            Layout.alignment: Qt.AlignHCenter
            font.pointSize:14
            font.family: webFont.name
            selectByMouse: true
            color: "#603140"}

        Button {
            Text {
                text: "Search"
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.verticalCenter: parent.verticalCenter
                font.bold: true
                font.family: webFont.name
                color: "#603140"}
            background: Rectangle {
                implicitWidth: 100
                implicitHeight: 40
                color: "white"}



            Layout.alignment: Qt.AlignHCenter
            onClicked: {
                weather.update_by_city(city_tf.text)
            }
        }

        Label{
            Layout.alignment: Qt.AlignHCenter
            id: temperature_lbl
            color: "white"
            font.family: webFont.name
            font.pointSize:14
            font { family: 'Helvetica'; pixelSize: 20; capitalization: Font.SmallCaps }
        }
        Label{
            Layout.alignment: Qt.AlignHCenter
            id: description_lbl
        }
        Item {
            Layout.fillHeight: true
        }
    }

    Connections {
        target: weather
        function onDataChanged(){
            if(!weather.hasError()){
                var temperature = weather.data['main']['temp']
                var main = weather.data['weather'][0]['main']
                var humidity = weather.data['main']['humidity']
                var wind_speed = weather.data['wind']['speed']
                temperature_lbl.text = " Temperature : " + Math.round(((temperature - 273.15) * 1.8) + 32) + " degrees Fahrenheit \n"  + " Current Conditions : " + main + "\n Humidity : " + humidity + "\n Wind Speed : " + wind_speed + " m/s"
                var sunrise = weather.data['sys']['sunrise']
                var sunset = weather.data['sys']['sunset']
                var now = weather.data['dt']
                if (now >= sunset)
                    image_tod.source = "night2.png"
                else if (now <= sunrise)
                    image_tod.source = "night2.png"
                else
                    image_tod.source = "day2.png"



            }
        }
    }

}